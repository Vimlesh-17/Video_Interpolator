import os
import torch
import cv2
import numpy as np
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from models.rife import RIFE
from models.ldmvfi import LDMVFI

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {
            'rife': self.load_model('rife'),
            'ldmvfi': self.load_model('ldmvfi')
        }
        
    def load_model(self, model_type):
        model = RIFE() if model_type == 'rife' else LDMVFI()
        model.load_state_dict(torch.load(f'weights/best_{model_type}.pth', 
                                    map_location=self.device))
        return model.eval().to(self.device)
    
    def process_video(self, input_path, output_path, model_type):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps*2, (854, 480))
        
        model = self.models[model_type]
        ret, prev_frame = cap.read()
        
        for frame_num in range(1, total_frames):
            ret, curr_frame = cap.read()
            if not ret: break
            
            # Process frames
            interpolated = self.interpolate_frames(prev_frame, curr_frame, model)
            out.write(prev_frame)
            out.write(interpolated)
            
            # Update progress
            progress = frame_num / total_frames * 100
            socketio.emit('progress', {'progress': progress})
            
            prev_frame = curr_frame
        
        cap.release()
        out.release()
    
    def interpolate_frames(self, frame1, frame2, model):
        # Preprocess
        t1 = self._preprocess_frame(frame1)
        t2 = self._preprocess_frame(frame2)

        # Inference
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(t1, t2)

        processed_frame = self._postprocess_output(output)
        return processed_frame

processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_video')
def handle_process(data):
    input_path = 'temp_input.mp4'
    output_path = 'temp_output.mp4'
    
    # Save uploaded file
    with open(input_path, 'wb') as f:
        f.write(data['video'])
    
    # Process video
    processor.process_video(input_path, output_path, data['model_type'])
    
    # Send result
    socketio.emit('complete', {'output_path': output_path})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)