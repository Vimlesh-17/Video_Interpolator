const socket = io();

function processVideo() {
    const file = document.getElementById('video-upload').files[0];
    const modelType = document.getElementById('model-select').value;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        socket.emit('process_video', {
            video: e.target.result,
            model_type: modelType
        });
    };
    reader.readAsArrayBuffer(file);
}

socket.on('progress', (data) => {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = `${data.progress}%`;
});

socket.on('complete', (data) => {
    const videoElement = document.createElement('video');
    videoElement.controls = true;
    videoElement.src = data.output_path;
    document.getElementById('output-video').appendChild(videoElement);
});