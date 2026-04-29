import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .splat_utils import logit, quat_normalize, sigmoid

PLY_TYPE_TO_STRUCT = {
    "char": "b", "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i",
    "uint": "I", "uint32": "I",
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
}
PLY_TYPE_TO_DTYPE = {
    "char": np.int8, "int8": np.int8,
    "uchar": np.uint8, "uint8": np.uint8,
    "short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
}

@dataclass
class PlyHeader:
    fmt: str            # "ascii" or "binary_little_endian"
    vertex_count: int
    properties: List[Tuple[str, str]]  # (ply_type, name)

def read_ply(path: str) -> tuple[
    PlyHeader,            # original header (incl. properties)
    np.ndarray,           # mu (N,3) float32, activated
    np.ndarray,           # op (N,) float32 in [0,1], activated
    np.ndarray,           # sc (N,3) float32 linear, activated
    np.ndarray,           # q  (N,4) float32 [w,x,y,z], normalized, activated
    np.ndarray,           # sh (N,C) float32 appearance (C can be 0), activated (raw passthrough)
    list[str],            # app_names (order matches sh columns)
]:
    """
    Read a PLY and return ACTIVATED gaussian attributes.

    Assumptions (RAW params in input PLY):
      - scale_* : raw log-scale (use exp() to get linear scale)
      - opacity : raw logit(alpha) (use sigmoid() to get alpha)
      - rot_*   : quaternion in [w,x,y,z] order (normalize before use)

    Appearance fields:
      - all float fields not in required/fixed and not in drop (stored as-is).
    """
    import struct

    required = [
        "x", "y", "z",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    drop = {"nx", "ny", "nz"}

    # -------- parse header + read raw columns --------
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF while reading PLY header.")
            header_lines.append(line.decode("ascii", errors="replace").rstrip("\n"))
            if header_lines[-1].strip() == "end_header":
                break

        fmt = None
        vertex_count = None
        props: list[tuple[str, str]] = []
        in_vertex = False

        for ln in header_lines:
            parts = ln.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                if parts[1] == "ascii":
                    fmt = "ascii"
                elif parts[1] == "binary_little_endian":
                    fmt = "binary_little_endian"
                else:
                    raise ValueError(f"Unsupported PLY format: {parts[1]}")
            elif parts[0] == "element":
                in_vertex = (parts[1] == "vertex")
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    continue
                pt, name = parts[1], parts[2]
                if pt not in PLY_TYPE_TO_DTYPE:
                    raise ValueError(f"Unsupported property type: {pt}")
                props.append((pt, name))

        if fmt is None or vertex_count is None:
            raise ValueError("Invalid PLY header: missing format/vertex count.")

        cols: dict[str, np.ndarray] = {}

        if fmt == "ascii":
            raw = f.read().decode("ascii", errors="replace").splitlines()
            if len(raw) < vertex_count:
                raise ValueError("Not enough vertex lines in ASCII PLY.")
            tmp = {name: [] for _, name in props}
            for i in range(vertex_count):
                parts = raw[i].strip().split()
                if len(parts) < len(props):
                    raise ValueError(f"Vertex line {i} has too few columns.")
                for (pt, name), val in zip(props, parts):
                    tmp[name].append(val)
            for pt, name in props:
                cols[name] = np.array(tmp[name], dtype=PLY_TYPE_TO_DTYPE[pt])
        else:
            fmt_chars = "".join(PLY_TYPE_TO_STRUCT[pt] for pt, _ in props)
            st = struct.Struct("<" + fmt_chars)
            buf = f.read(st.size * vertex_count)
            if len(buf) < st.size * vertex_count:
                raise ValueError("Not enough binary vertex bytes.")
            dtype_fields = [(name, PLY_TYPE_TO_DTYPE[pt]) for pt, name in props]
            dt = np.dtype(dtype_fields)
            arr = np.frombuffer(buf, dtype=dt, count=vertex_count)
            for _, name in props:
                cols[name] = arr[name].copy()

    hdr = PlyHeader(fmt, vertex_count, props)

    # -------- validate + decide appearance fields --------
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required property '{r}'.")

    fixed = set(required) | {"x", "y", "z"}
    app_names: list[str] = []
    for pt, name in hdr.properties:
        if name in drop or name in fixed:
            continue
        if pt in ("float", "float32", "double", "float64"):
            app_names.append(name)

    # -------- build ACTIVATED arrays --------
    mu = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float32)

    op_raw = cols["opacity"].astype(np.float32)  # raw logit(alpha)
    sc_raw = np.stack(
        [cols["scale_0"], cols["scale_1"], cols["scale_2"]], axis=1
    ).astype(np.float32)  # raw log-scale
    q_raw = np.stack(
        [cols["rot_0"], cols["rot_1"], cols["rot_2"], cols["rot_3"]], axis=1
    ).astype(np.float32)  # raw quat [wxyz]

    if app_names:
        sh = np.stack([cols[n].astype(np.float32) for n in app_names], axis=1)
    else:
        sh = np.zeros((mu.shape[0], 0), dtype=np.float32)

    # activate
    op = sigmoid(op_raw).astype(np.float32)
    sc = np.exp(np.clip(sc_raw, -30.0, 30.0)).astype(np.float32)
    q = quat_normalize(q_raw).astype(np.float32)

    return hdr, mu, op, sc, q, sh, app_names

def store_ply(
        out_path: str,
        hdr: PlyHeader,
        mu: np.ndarray,
        op: np.ndarray,
        sc: np.ndarray,
        q: np.ndarray,
        sh: np.ndarray,
        app_names: List[str],
    ) -> None:
    op_raw_out = logit(op).astype(np.float32)
    sc_raw_out = np.log(np.maximum(sc, 1e-12)).astype(np.float32)

    out_props = [(pt, name) for pt, name in hdr.properties]
    out_cols: Dict[str, np.ndarray] = {}
    app_to_col = {name: idx for idx, name in enumerate(app_names)}

    for pt, name in out_props:
        dt = PLY_TYPE_TO_DTYPE[pt]
        if name == "x":
            out_cols[name] = mu[:, 0].astype(dt, copy=False)
        elif name == "y":
            out_cols[name] = mu[:, 1].astype(dt, copy=False)
        elif name == "z":
            out_cols[name] = mu[:, 2].astype(dt, copy=False)
        elif name == "opacity":
            out_cols[name] = op_raw_out.astype(dt, copy=False)
        elif name.startswith("scale_"):
            k = int(name.split("_")[1])
            out_cols[name] = sc_raw_out[:, k].astype(dt, copy=False)
        elif name.startswith("rot_"):
            k = int(name.split("_")[1])
            out_cols[name] = q[:, k].astype(dt, copy=False)  # [w,x,y,z]
        else:
            if name in app_to_col:
                out_cols[name] = sh[:, app_to_col[name]].astype(dt, copy=False)
            else:
                out_cols[name] = np.zeros(mu.shape[0], dtype=dt)

    write_ply_binary_little_endian(out_path, out_props, out_cols)

def _fmt_ascii(v) -> str:
    if isinstance(v, (np.floating, float)):
        return f"{float(v):.8g}"
    return str(int(v))

def write_ply_ascii(path: str, properties: List[Tuple[str, str]], cols: Dict[str, np.ndarray]) -> None:
    n = len(next(iter(cols.values())))
    names = [name for _, name in properties]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for pt, name in properties:
            f.write(f"property {pt} {name}\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(" ".join(_fmt_ascii(cols[name][i]) for name in names) + "\n")

def write_ply_binary_little_endian(path: str, properties: List[Tuple[str, str]], cols: Dict[str, np.ndarray]) -> None:
    """
    Writes vertex-only PLY as binary_little_endian with given property order/types.
    """
    n = len(next(iter(cols.values())))
    names = [name for _, name in properties]
    with open(path, "wb") as f:
        header = []
        header.append("ply")
        header.append("format binary_little_endian 1.0")
        header.append(f"element vertex {n}")
        for pt, name in properties:
            header.append(f"property {pt} {name}")
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))

        fmt_chars = "".join(PLY_TYPE_TO_STRUCT[pt] for pt, _ in properties)
        st = struct.Struct("<" + fmt_chars)

        # Ensure contiguous arrays and correct dtypes
        arrays = []
        for pt, name in properties:
            dt = PLY_TYPE_TO_DTYPE[pt]
            arr = np.asarray(cols[name])
            if arr.dtype != dt:
                arr = arr.astype(dt, copy=False)
            arrays.append(np.ascontiguousarray(arr))

        # Pack row-by-row (fast enough; avoids huge intermediate structured arrays)
        # If you need more speed, you can build a structured array and tofile().
        for i in range(n):
            row = [arrays[k][i] for k in range(len(arrays))]
            # Convert numpy scalars to python scalars for struct
            row = [x.item() if isinstance(x, np.generic) else x for x in row]
            f.write(st.pack(*row))

