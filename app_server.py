from concurrent.futures import ProcessPoolExecutor
from flask import Flask, request, jsonify
from gradio import Interface
import os
from app import concatenate_images
import cv2
import time
from werkzeug.utils import secure_filename

from app_try import Trainer
from facechain.inference import GenPortrait

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
HISTORY_FOLDER = 'history/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)


def train_model(files, output_model_name):
    trainer = Trainer()
    instance_images = []

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        instance_images.append(img)

    model_path = trainer.run("", "AI-ModelScope/stable-diffusion-v1-5", instance_images, output_model_name)
    return model_path

@app.route('/train', methods=['POST'])
def train():
    files = request.files.getlist("files")
    output_model_name = request.form.get("output_model_name")
    
    if not files or not output_model_name:
        return jsonify({"error": "No files or output_model_name provided"}), 400

    model_path = train_model(files, output_model_name)
    return jsonify({"model_path": model_path})

@app.route('/get_uploaded_images', methods=['GET'])
def get_uploaded_images():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], file) for file in files]
    return jsonify(file_paths)


def generate_image(state, base_model_index, user_model, num_images, lora_choice, style_model, multiplier_style, multiplier_human, sr_img_size, cartoon_style_idx, use_lcm_idx):

    gen_portrait = GenPortrait(pose_model_path=None, pose_image=None, use_depth_control=False, pos_prompt="", neg_prompt="", style_model_path='style_model_path', 
                               multiplier_style=multiplier_style, multiplier_human=multiplier_human, use_main_model=True,
                               use_face_swap=False, use_post_process=False,
                               use_stylization=False)

    num_images = min(6, num_images)
    state['inference_done_count'] = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        future = executor.submit(gen_portrait, 'instance_data_dir',
                                              num_images, 'base_model', 'lora_model_path', 'sub_path', 'revision', sr_img_size, cartoon_style_idx, use_lcm_idx=use_lcm_idx)
        while not future.done():
            is_processing = future.running()
            if not is_processing:
                cur_done_count = state['inference_done_count']
                to_wait = state['before_queue_size'] - (cur_done_count - state['before_done_count'])
                yield [state, 'infer_choose_block', 'inference_result_block', f"排队等待资源中, 前方还有{to_wait}个生成任务, 预计需要等待{to_wait * 2.5}分钟...",
                        None]
            else:
                yield [state, 'infer_choose_block', 'inference_result_block', "正在为你生成中，请耐心等待...", None]
            time.sleep(1)

    outputs = future.result()
    outputs_RGB = [cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB) for out_tmp in outputs]
    
    save_dir = os.path.join('data', uuid, 'inference_result', 'base_model', user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # use single to save outputs
    if not os.path.exists(os.path.join(save_dir, 'single')):
        os.makedirs(os.path.join(save_dir, 'single'))
    for img in outputs:
        # count the number of images in the folder
        num = len(os.listdir(os.path.join(save_dir, 'single')))
        cv2.imwrite(os.path.join(save_dir, 'single', str(num) + '.png'), img)
    
    if len(outputs) > 0:
        result = concatenate_images(outputs)
        if not os.path.exists(os.path.join(save_dir, 'concat')):
            os.makedirs(os.path.join(save_dir, 'concat'))
        num = len(os.listdir(os.path.join(save_dir, 'concat')))
        image_path = os.path.join(save_dir, 'concat', str(num) + '.png')
        cv2.imwrite(image_path, result)

        yield [state, 'infer_choose_block', 'inference_result_block', "生成完成，可点击喜欢的造型进行查看或保存。", outputs_RGB]
    else:
        yield [state, 'infer_choose_block', 'inference_result_block', "生成失败, 请重试!", outputs_RGB]


def upload_file(files, current_images):
    current_images = current_images or []
    saved_files = []

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        saved_files.append(file_path)

    if len(current_images) + len(saved_files) >= 3:
        return jsonify({
            "current_images": current_images + saved_files
        })
    else:
        return jsonify({
            "current_images": current_images + saved_files
        })
    
def load_all_history(uuid, base_model_index=None, user_model=None, lora_choice=None, style_model=None):
    base_model = 'base_model'  # 示例值，需要根据实际逻辑替换
    save_dir = os.path.join(HISTORY_FOLDER, uuid, 'inference_result', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])
    
    if not os.path.exists(save_dir):
        return [], []
    
    single_dir = os.path.join(save_dir, 'single')
    concat_dir = os.path.join(save_dir, 'concat')
    single_imgs = []
    concat_imgs = []
    if os.path.exists(single_dir):
        single_imgs = sorted(os.listdir(single_dir))
        single_imgs = [(os.path.join(single_dir, img), f"{user_model} {style_model}") for img in single_imgs]
    if os.path.exists(concat_dir):
        concat_imgs = sorted(os.listdir(concat_dir))
        concat_imgs = [(os.path.join(concat_dir, img), f"{user_model} {style_model}") for img in concat_imgs]
    return single_imgs, concat_imgs

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = generate_image(data['state'], data['uuid'], data['pos_prompt'], data['neg_prompt'], data['base_model_index'], data['user_model'], data['num_images'], data['lora_choice'], data['style_model'], data['multiplier_style'], data['multiplier_human'], data['pose_model'], data['pose_image'], data['sr_img_size'], data['cartoon_style_idx'], data['use_lcm_idx'])
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist("files")
    current_images = request.form.getlist("current_images")
    hint = request.form.get("hint", "Upload at least 3 images.")

    response = upload_file(files, current_images, hint)
    return response

@app.route('/get_history_images', methods=['GET'])
def get_history_images():
    uuid = request.args.get('uuid', 'qw')
    base_model_index = request.args.get('base_model_index')
    user_model = request.args.get('user_model')
    lora_choice = request.args.get('lora_choice')
    style_model = request.args.get('style_model')

    single_imgs, concat_imgs = load_all_history(uuid, base_model_index, user_model, lora_choice, style_model)
    return jsonify({
        'single_imgs': single_imgs,
        'concat_imgs': concat_imgs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
