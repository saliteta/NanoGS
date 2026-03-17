document.addEventListener('DOMContentLoaded', domReady);
    let dicsRealWorld = null;
    let dicsStatic = null;
    let dicsDynamic = null;
    let dicsDemo = null;
        function domReady() {
            dicsRealWorld = new Dics({
                container: document.querySelectorAll('.b-dics.real-world')[0],
                hideTexts: false,
                textPosition: "bottom"
            });
            dicsDemo = new Dics({
                container: document.querySelectorAll('.b-dics.demo')[0],
                hideTexts: false,
                textPosition: "bottom"
            });
            dicsStatic = new Dics({
                container: document.querySelectorAll('.b-dics.static')[0],
                hideTexts: false,
                textPosition: "bottom"
            });
            dicsDynamic = new Dics({
                container: document.querySelectorAll('.b-dics.dynamic')[0],
                hideTexts: false,
                textPosition: "bottom"
            });
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

