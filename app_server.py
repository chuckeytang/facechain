from concurrent.futures import ProcessPoolExecutor
from flask import Flask, request, jsonify
from gradio import Interface
import os
import cv2
import time

from app_try import Trainer
from facechain.inference import GenPortrait

app = Flask(__name__)

# 定义你的Gradio接口函数
def train_model(uuid, base_model_name, instance_images, output_model_name):
    trainer = Trainer()
    model_path = trainer.run(uuid, base_model_name, instance_images, output_model_name)
    return model_path

def generate_image(state, uuid, pos_prompt, neg_prompt, base_model_index, user_model, num_images, lora_choice, style_model, multiplier_style, multiplier_human, pose_model, pose_image, sr_img_size, cartoon_style_idx, use_lcm_idx):
    gen_portrait = GenPortrait(pose_model_path='pose_model_path', pose_image=pose_image, use_depth_control=False, pos_prompt=pos_prompt, neg_prompt=neg_prompt, style_model_path='style_model_path', 
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


# 创建API端点
@app.route('/train', methods=['POST'])
def train():
    data = request.json
    result = train_model(data['uuid'], data['base_model_name'], data['instance_images'], data['output_model_name'])
    return jsonify(result)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = generate_image(data['state'], data['uuid'], data['pos_prompt'], data['neg_prompt'], data['base_model_index'], data['user_model'], data['num_images'], data['lora_choice'], data['style_model'], data['multiplier_style'], data['multiplier_human'], data['pose_model'], data['pose_image'], data['sr_img_size'], data['cartoon_style_idx'], data['use_lcm_idx'])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
