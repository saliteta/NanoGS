# NanoGS: Training-Free Gaussian Splat Simplification

<p align="center">
  <!-- arXiv -->
  <a href="https://arxiv.org/abs/2603.16103">
    <img src="https://img.shields.io/badge/arXiv-2603.16103-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <!-- Paper PDF -->
  <a href="https://arxiv.org/pdf/2603.16103">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge&logo=adobeacrobatreader" alt="Paper PDF">
  </a>
  <!-- Project page -->
  <a href="https://saliteta.github.io/NanoGS/">
    <img src="https://img.shields.io/badge/Project-Page-1abc9c?style=for-the-badge&logo=googlechrome" alt="Project page">
  </a>
  <!-- Web app -->
  <a href="https://rongliu-leo.github.io/NanoGS/">
    <img src="https://img.shields.io/badge/Web-App-3498db?style=for-the-badge&logo=googlechrome" alt="Web App">
  </a>
  <!-- Code (Web App) -->
  <a href="https://github.com/RongLiu-Leo/NanoGS">
    <img src="https://img.shields.io/badge/Code-Web-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Code (Web App)">
  </a>
</p>

---


<p align="center">
  <img src="teaser.png" alt="NanoGS teaser" width="80%">
</p>

<p align="center">
  <em>NanoGS achieves substantial simplification ratios while preserving fidelity and geometry—without post-optimization or calibrated images.</em>
</p>


## How to use

Run the simplifier on a 3DGS `.ply` file:

```bash
python simplification.py --ply <input.ply> [-o <output.ply>] [-r RATIO] [--k K] [--merge_cap MERGE_CAP] [--opacity_threshold OPACITY_THRESHOLD] [--lam_geo LAM_GEO] [--lam_sh LAM_SH]
```

**Required**

| Option | Description |
|--------|-------------|
| `--ply PLY` | Input PLY file (3DGS ply file). |

**Optional**

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | auto from input + ratio | Output PLY path. |
| `-r`, `--ratio` | `0.5` | Fraction of splats to keep, in (0,1). Example: `0.25` keeps 25%. |
| `--k` | `16` | k for KNN candidates. |
| `--merge_cap` | `0.5` | Max merges per pass as ratio of original splat count (0.01–0.5). |
| `--opacity_threshold` | `0.1` | Prune splats with opacity below this before merging. The effective threshold is `min(threshold, median(opacity))`. |
| `--lam_geo` | `1.0` | Geometry term weight in merge cost. |
| `--lam_sh` | `1.0` | Spherical-harmonics term weight in merge cost. |

**Examples**

```bash
# Keep 25% of splats, output auto-named (e.g. scene_0.25.ply)
python simplification.py --ply scene.ply -r 0.25

# Custom output and all optional parameters
python simplification.py --ply scene.ply -o simplified.ply -r 0.5 --k 16 --merge_cap 0.3 --opacity_threshold 0.1 --lam_geo 1.0 --lam_sh 1.0
```

## Citation
If you find our code or paper helps, please consider giving us a star or citing:
```bibtex
@misc{xiong2026nanogstrainingfreegaussiansplat,
    title={NanoGS: Training-Free Gaussian Splat Simplification}, 
    author={Butian Xiong and Rong Liu and Tiantian Zhou and Meida Chen and Zhiwen Fan and Andrew Feng},
    year={2026},
    eprint={2603.16103},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.16103}, 
}
}
```

## License

See `LICENSE` in this repository.
