document.addEventListener('DOMContentLoaded', domReady);
    let dicsRealWorld = null;
    let dicsStatic = null;
    let dicsDynamic = null;
    let dicsDemo = null;
    let dicsCompressionReal = null;
    let dicsCompressionSynthetic = null;

    const COMPRESSION_SCENES_REAL = ['bicycle', 'bonsai', 'counter', 'drjohnson', 'flowers', 'garden', 'kitchen', 'playroom', 'room', 'stump', 'train', 'treehill', 'truck'];
    const COMPRESSION_SCENES_SYNTHETIC = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'];
    const COMPRESSION_RATIO_LABELS = ['100%', '10%', '1%', '0.1%'];
    const COMPRESSION_RATIO_FOLDERS = [null, '0_1', '0_01', '0_001'];
    let compressionRealSceneIdx = 0;
    let compressionRealRatioIdx = 1;
    let compressionSyntheticSceneIdx = 0;
    let compressionSyntheticRatioIdx = 1;

    function compressionGtImageSrc(scene) {
        return './static/images/compression/' + scene + '/gt.png';
    }

    function compressionMethodImageSrc(scene, ratioFolder, method) {
        return './static/images/compression/' + scene + '/' + method + '_' + ratioFolder + '.png';
    }

    function compressionInjectGt(container, scene) {
        if (!container) return;
        container.innerHTML = '';
        container.classList.add('b-dics--tp-bottom');
        const mediaContainer = document.createElement('div');
        mediaContainer.className = 'b-dics__media-container';
        const img = document.createElement('img');
        img.src = compressionGtImageSrc(scene);
        img.alt = 'Ground Truth';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'contain';
        img.style.display = 'block';
        const label = document.createElement('p');
        label.className = 'b-dics__text';
        label.textContent = 'Ground Truth';
        mediaContainer.appendChild(img);
        mediaContainer.appendChild(label);
        container.appendChild(mediaContainer);
    }

    function compressionInjectCompare(container, sceneList, sceneIdx, ratioIdx) {
        if (!container || ratioIdx === 0) return;
        const scene = sceneList[sceneIdx];
        const ratioFolder = COMPRESSION_RATIO_FOLDERS[ratioIdx];
        const labels = ['LightGS', 'PUP3DGS', 'GHAP', 'Ours'];
        const methods = ['light', 'pup', 'gmm', 'ours'];
        container.innerHTML = '';
        for (let i = 0; i < 4; i++) {
            const img = document.createElement('img');
            img.src = compressionMethodImageSrc(scene, ratioFolder, methods[i]);
            img.alt = labels[i];
            container.appendChild(img);
        }
    }

    function compressionUpdateRealDisplay() {
        const idx = compressionRealRatioIdx;
        const gtEl = document.getElementById('compression-real-gt');
        const compareEl = document.getElementById('compression-real-compare');
        const scene = COMPRESSION_SCENES_REAL[compressionRealSceneIdx];
        const refHeight = Math.max(
            gtEl && gtEl.offsetHeight ? gtEl.offsetHeight : 0,
            compareEl && compareEl.offsetHeight ? compareEl.offsetHeight : 0
        );
        if (idx === 0) {
            if (gtEl) { gtEl.style.display = ''; const img = gtEl.querySelector('.b-dics__media-container img'); if (img) img.src = compressionGtImageSrc(scene); }
            if (compareEl) compareEl.style.display = 'none';
        } else {
            if (gtEl) gtEl.style.display = 'none';
            if (compareEl) {
                compareEl.style.display = '';
                const imgs = compareEl.querySelectorAll('img');
                const ratioFolder = COMPRESSION_RATIO_FOLDERS[idx];
                const methods = ['light', 'pup', 'gmm', 'ours'];
                for (let i = 0; i < 4; i++) if (imgs[i]) imgs[i].src = compressionMethodImageSrc(scene, ratioFolder, methods[i]);
                if (dicsCompressionReal) dicsCompressionReal.medias = dicsCompressionReal._getMedias();
            }
        }
        const h = refHeight;
        if (h > 0) {
            if (gtEl) gtEl.style.minHeight = h + 'px';
            if (compareEl) compareEl.style.minHeight = h + 'px';
        }
    }

    function compressionUpdateSyntheticDisplay() {
        const idx = compressionSyntheticRatioIdx;
        const gtEl = document.getElementById('compression-synthetic-gt');
        const compareEl = document.getElementById('compression-synthetic-compare');
        const scene = COMPRESSION_SCENES_SYNTHETIC[compressionSyntheticSceneIdx];
        const refHeight = Math.max(
            gtEl && gtEl.offsetHeight ? gtEl.offsetHeight : 0,
            compareEl && compareEl.offsetHeight ? compareEl.offsetHeight : 0
        );
        if (idx === 0) {
            if (gtEl) { gtEl.style.display = ''; const img = gtEl.querySelector('.b-dics__media-container img'); if (img) img.src = compressionGtImageSrc(scene); }
            if (compareEl) compareEl.style.display = 'none';
        } else {
            if (gtEl) gtEl.style.display = 'none';
            if (compareEl) {
                compareEl.style.display = '';
                const imgs = compareEl.querySelectorAll('img');
                const ratioFolder = COMPRESSION_RATIO_FOLDERS[idx];
                const methods = ['light', 'pup', 'gmm', 'ours'];
                for (let i = 0; i < 4; i++) if (imgs[i]) imgs[i].src = compressionMethodImageSrc(scene, ratioFolder, methods[i]);
                if (dicsCompressionSynthetic) dicsCompressionSynthetic.medias = dicsCompressionSynthetic._getMedias();
            }
        }
        const h = refHeight;
        if (h > 0) {
            if (gtEl) gtEl.style.minHeight = h + 'px';
            if (compareEl) compareEl.style.minHeight = h + 'px';
        }
    }

    function compressionRealSceneEvent(idx) {
        compressionRealSceneIdx = idx;
        compressionUpdateRealDisplay();
        const list = document.getElementById('compression-real-scene').children;
        for (let i = 0; i < list.length; i++) list[i].children[0].className = (idx === i) ? 'nav-link active' : 'nav-link';
    }

    function compressionRealRatioSlider(el) {
        const idx = parseInt(el.value, 10);
        compressionRealRatioIdx = idx;
        const labelEl = document.getElementById('compression-real-ratio-label');
        if (labelEl) labelEl.textContent = COMPRESSION_RATIO_LABELS[idx];
        compressionUpdateRealDisplay();
    }

    function compressionSyntheticSceneEvent(idx) {
        compressionSyntheticSceneIdx = idx;
        compressionUpdateSyntheticDisplay();
        const list = document.getElementById('compression-synthetic-scene').children;
        for (let i = 0; i < list.length; i++) list[i].children[0].className = (idx === i) ? 'nav-link active' : 'nav-link';
    }

    function compressionSyntheticRatioSlider(el) {
        const idx = parseInt(el.value, 10);
        compressionSyntheticRatioIdx = idx;
        const labelEl = document.getElementById('compression-synthetic-ratio-label');
        if (labelEl) labelEl.textContent = COMPRESSION_RATIO_LABELS[idx];
        compressionUpdateSyntheticDisplay();
    }

        function domReady() {
            const realGtEl = document.getElementById('compression-real-gt');
            const realCompareEl = document.getElementById('compression-real-compare');
            const syntheticGtEl = document.getElementById('compression-synthetic-gt');
            const syntheticCompareEl = document.getElementById('compression-synthetic-compare');
            compressionInjectGt(realGtEl, COMPRESSION_SCENES_REAL[compressionRealSceneIdx]);
            compressionInjectGt(syntheticGtEl, COMPRESSION_SCENES_SYNTHETIC[compressionSyntheticSceneIdx]);
            compressionInjectCompare(realCompareEl, COMPRESSION_SCENES_REAL, compressionRealSceneIdx, compressionRealRatioIdx);
            compressionInjectCompare(syntheticCompareEl, COMPRESSION_SCENES_SYNTHETIC, compressionSyntheticSceneIdx, compressionSyntheticRatioIdx);
            if (realGtEl) realGtEl.style.display = 'none';
            if (syntheticGtEl) syntheticGtEl.style.display = 'none';
            document.getElementById('compression-real-ratio-label').textContent = COMPRESSION_RATIO_LABELS[compressionRealRatioIdx];
            document.getElementById('compression-synthetic-ratio-label').textContent = COMPRESSION_RATIO_LABELS[compressionSyntheticRatioIdx];
            if (realCompareEl) {
                dicsCompressionReal = new Dics({
                    container: realCompareEl,
                    hideTexts: false,
                    textPosition: "bottom"
                });
            }
            if (syntheticCompareEl) {
                dicsCompressionSynthetic = new Dics({
                    container: syntheticCompareEl,
                    hideTexts: false,
                    textPosition: "bottom"
                });
            }
            compressionUpdateRealDisplay();
            compressionUpdateSyntheticDisplay();
            const rw = document.querySelectorAll('.b-dics.real-world')[0];
            if (rw) dicsRealWorld = new Dics({ container: rw, hideTexts: false, textPosition: "bottom" });
            const demo = document.querySelectorAll('.b-dics.demo')[0];
            if (demo) dicsDemo = new Dics({ container: demo, hideTexts: false, textPosition: "bottom" });
            const st = document.querySelectorAll('.b-dics.static')[0];
            if (st) dicsStatic = new Dics({ container: st, hideTexts: false, textPosition: "bottom" });
            const dyn = document.querySelectorAll('.b-dics.dynamic')[0];
            if (dyn) dicsDynamic = new Dics({ container: dyn, hideTexts: false, textPosition: "bottom" });
        }

        function realWorldVideoEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.real-world')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];

                let parts = media.src.split('/');

                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'coffee_martini';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'cut_roasted_beef';
                        break;
                    case 2:
                        parts[parts.length - 2] = 'flame_steak';
                        break;
                    case 3:
                        parts[parts.length - 2] = 'sear_steak';
                        break;
                }

                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }
            let scene_list = document.getElementById("real-world-video").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsRealWorld.medias = dicsRealWorld._getMedias();
        }

        function demoVideoEvent(idx) {
            let video = document.querySelectorAll('.b-dics.demo')[0].getElementsByTagName('video')[0];
            let parts = video.children[0].src.split('/');
            switch (idx) {
                case 0:
                    parts[parts.length - 1] = 'dynamic.mp4';
                    break;
                case 1:
                    parts[parts.length - 1] = 'renders.mp4';
                    break;
                case 2:
                    parts[parts.length - 1] = 'temporal.mp4';
                    break;
                case 3:
                    parts[parts.length - 1] = 'angular.mp4';
                    break;
                case 4:
                    parts[parts.length - 1] = 'spatial.mp4';
                    break;
            }
            let newSrc = parts.join('/');
            let newVideo = video.cloneNode(true);
            newVideo.children[0].src = newSrc;
            video.parentNode.replaceChild(newVideo, video);

            let video_list = document.getElementById("demo-video").children;
            for (let i = 0; i < video_list.length; i++) {
                if (idx == i) {
                    video_list[i].children[0].className = "nav-link active"
                }
                else {
                    video_list[i].children[0].className = "nav-link"
                }
            }
            dicsDemo.medias = dicsDemo._getMedias();
        }

        function staticSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.static')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'counter';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'bicycle';
                        break;
                    case 2:
                        parts[parts.length - 2] = 'bonsai';
                        break;
                    case 3:
                        parts[parts.length - 2] = 'kitchen';
                        break;
                    case 4:
                        parts[parts.length - 2] = 'smoke';
                        break;
                    case 5:
                        parts[parts.length - 2] = 'explosion';
                        break;
                    case 6:
                        parts[parts.length - 2] = 'bunny';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("static-reconstruction").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsStatic.medias = dicsStatic._getMedias();
        }


        function dynamicSceneEvent(idx) {
            let sections = document.querySelectorAll('.b-dics.dynamic')[0].getElementsByClassName('b-dics__section')
            for (let i = 0; i < sections.length; i++) {
                let mediaContainer = sections[i].getElementsByClassName('b-dics__media-container')[0];
                let media = mediaContainer.getElementsByClassName('b-dics__media')[0];
        
                let parts = media.src.split('/');
        
                switch (idx) {
                    case 0:
                        parts[parts.length - 2] = 'cloud';
                        break;
                    case 1:
                        parts[parts.length - 2] = 'heart';
                        break;
                    case 2:
                        parts[parts.length - 2] = 'suzanne';
                        break;
                    case 3:
                        parts[parts.length - 2] = 'trex';
                        break;
                }
        
                let newSrc = parts.join('/');
                let newMedia = media.cloneNode(true);
                newMedia.src = newSrc;
                mediaContainer.replaceChild(newMedia, media);
            }

            let scene_list = document.getElementById("dynamic-reconstruction").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
            dicsDynamic.medias = dicsDynamic._getMedias();
        }

