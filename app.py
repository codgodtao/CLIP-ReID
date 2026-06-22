"""CLIP-ReID Attention 可视化 Web 界面

基于 Flask 的简单 Web 应用，支持：
1. 上传图像
2. 生成 Attention 热力图
3. 可视化展示
"""
import os
import io
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image
import torch
import sys

# Add project path
sys.path.insert(0, '/workspace')

from visualization.attention_viz import generate_attention_visualization
from config import cfg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/workspace/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Global model cache
model = None


def load_model():
    """Load CLIP-ReID model once at startup."""
    global model
    if model is None:
        from model.make_model_mobileclip2 import build_transformer
        
        cfg.merge_from_file('/workspace/configs/person/vit_mobileclip2.yml')
        cfg.freeze()
        
        model = build_transformer(num_classes=1000, camera_num=1, view_num=1, cfg=cfg)
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        print("Model loaded successfully")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Load image
            img = Image.open(file.stream).convert('RGB')
            
            # Generate attention visualization
            try:
                results = generate_attention_visualization(model, img)
                
                # Save results
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                result_paths = {}
                
                for layer_name, vis_img in results.items():
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{layer_name}.png')
                    vis_img.save(save_path)
                    result_paths[layer_name] = f'/result/{layer_name}'
                
                return jsonify({
                    'success': True,
                    'layers': list(result_paths.keys()),
                    'paths': result_paths
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    # GET request - show form
    return render_template('index.html')


@app.route('/result/<layer_name>')
def get_result(layer_name):
    """Serve attention visualization result."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{layer_name}.png')
    
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "File not found", 404


@app.route('/api/visualize', methods=['POST'])
def api_visualize():
    """API endpoint for attention visualization."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    
    try:
        results = generate_attention_visualization(model, img)
        layer_names = list(results.keys())
        
        # Save and return paths
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        paths = {}
        
        for layer_name, vis_img in results.items():
            filename = f'{layer_name}_{hash(layer_name)}.png'
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            vis_img.save(save_path)
            paths[layer_name] = f'/result/{layer_name}'
        
        return jsonify({
            'success': True,
            'message': 'Visualization generated successfully',
            'layers': layer_names,
            'paths': paths
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model before starting server
    load_model()
    
    # Create templates folder
    os.makedirs('/workspace/templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)