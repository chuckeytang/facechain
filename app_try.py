import gradio as gr
import queue
from app import concatenate_images, train_lora_fn, generate_pos_prompt
import os
import torch
import slugify
import shutil
import json
import time
import cv2

from concurrent.futures import ProcessPoolExecutor
from facechain.utils import snapshot_download, set_spawn_method, project_dir, join_worker_data_dir
from facechain.constants import neg_prompt as neg, pose_models, base_models, pose_examples
from facechain.train_text_to_image_lora import prepare_dataset, data_process_fn
from facechain.inference import GenPortrait

training_done_count = 0
inference_done_count = 0
SDXL_BASE_MODEL_ID = 'AI-ModelScope/stable-diffusion-xl-base-1.0'
character_model = 'ly261666/cv_portrait_model'
BASE_MODEL_MAP = {
    "leosamsMoonfilm_filmGrain20": "写实模型",
    "MajicmixRealistic_v6": "写真模型"
}
IS_TRAINING = False
init_hint_content = """
        1. 请上传3~10张头肩照，照片越多效果越好哦
        2. 避免图片中出现多人脸、脸部遮挡等情况，否则可能导致效果异常
        3. 请等待上传图片加载显示出来再点，否则会报错
        4. 双击图片可移除
        """

to_train_hint_content = """
            1. 请等待上传图片全部加载显示后，再点击“训练模型分身”按钮
            2. 模型训练中，形象定制模型训练中每张图片需约1.5分钟，请耐心等待
            3. 模型训练过程请勿刷新或关闭页面
            4. 双击图片可移除
            5. 训练大约需要5-20分钟，无须等待，可于一段时间后重新打开应用查看训练结果
            """


class Trainer:
    def __init__(self):
        # threading.Thread(target=self.process_queue, daemon=True).start()
        pass

    # 创建全局任务队列
    task_queue = queue.Queue()

    def run(self, uuid: str, base_model_name: str, instance_images: list, output_model_name: str) -> str:
        global IS_TRAINING
        # Check Training
        print(f"IS_TRAINING {IS_TRAINING}")
        if IS_TRAINING:
            raise gr.Error('服务器正在进行其他训练，请稍后再试')

        try:
            IS_TRAINING = True
            print(f"IS_TRAINING_SET {IS_TRAINING}")
            # time.sleep(120)
            set_spawn_method()

            # Check Cuda
            if not torch.cuda.is_available():
                raise gr.Error('服务器显卡不可用')

            # Check Cuda Memory
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                required_memory_bytes = 18 * (1024 ** 3)  # 18GB
                try:
                    # create 18GB tensor to check the memory if enough
                    tensor = torch.empty((required_memory_bytes // 4,), device=device)
                    print("显存足够")
                    del tensor
                except RuntimeError as e:
                    raise gr.Error("目前显存不足18GB，训练失败！")

            # Check Instance Valid
            if instance_images is None:
                raise gr.Error('请选择照片上传!')

            # Check output model name
            if not output_model_name:
                raise gr.Error('请输入模型分身名称')

            # Limit input Image
            if len(instance_images) > 20:
                raise gr.Error('请最多上传20张训练图片')

            # Check UUID & Studio
            if not uuid:
                if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
                    return "请登陆后使用! "
                else:
                    uuid = 'qw'
            if base_model_name == SDXL_BASE_MODEL_ID:
                print('** Setting base model to SDXL **')
                base_model_path = SDXL_BASE_MODEL_ID
                revision = 'v1.0.9'
                sub_path = ''
            else:
                print('** Setting base model to SD1.5 **')
                base_model_path = 'ly261666/cv_portrait_model'
                revision = 'v2.0'
                sub_path = "film/film"

            output_model_name = slugify.slugify(output_model_name)

            # mv user upload data to target dir
            instance_data_dir = join_worker_data_dir(uuid, 'training_data', base_model_path, output_model_name)
            print("--------uuid: ", uuid)

            uuid_dir = join_worker_data_dir(uuid)
            if not os.path.exists(uuid_dir):
                os.makedirs(uuid_dir)
            work_dir = join_worker_data_dir(uuid, base_model_path, output_model_name)

            if os.path.exists(work_dir):
                raise gr.Error("模型分身名称已存在，请换一个")

            print("----------work_dir: ", work_dir)
            shutil.rmtree(work_dir, ignore_errors=True)
            shutil.rmtree(instance_data_dir, ignore_errors=True)

            prepare_dataset([img[0] for img in instance_images], output_dataset_dir=instance_data_dir)
            data_process_fn(instance_data_dir, True)

            # train lora
            print("instance_data_dir", instance_data_dir)

            train_lora_fn(base_model_path=base_model_path,
                          revision=revision,
                          sub_path=sub_path,
                          output_img_dir=instance_data_dir,
                          work_dir=work_dir,
                          photo_num=len(instance_images))

            message = '''<center>**训练完成，请回到首页生成自己的AI写真**</center>'''
            print(message)

            run_button = gr.Button(visible=False)
            upload_button = gr.Button(visible=False)
            clear_button = gr.Button(visible=False)
            inference_button1 = gr.Button(visible=True)

            user_model_list = update_output_model()
            print(f"chuckeytang user_model_list: {user_model_list}")
            return run_button, upload_button, clear_button, inference_button1, gr.Markdown(value=message), gr.Gallery(
                value=user_model_list, allow_preview=False, interactive=False, label="分身")
        finally:
            IS_TRAINING = False


def get_model_list():
    style_list = base_models[1]['style_list']

    sub_styles = []
    for style in style_list:
        matched = list(filter(lambda item: style == item['name'], styles))
        sub_styles.append(matched[0])

    return sub_styles


def deal_history_inner(uuid, base_model_index=None, user_model=None, lora_choice=None, style_model=None,
                       deal_type="load"):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'

    if deal_type == "update":
        if (base_model_index is None) or (user_model is None) or (lora_choice is None) or (
                style_model is None and lora_choice == 'preset'):
            return gr.Gallery(value=[], visible=True), gr.Gallery(value=[],
                                                                  visible=True)  # error triggered by option change, won't pop up warning

    if deal_type == "load":
        return load_all_history(uuid, base_model_index, user_model, lora_choice, style_model)

    if base_model_index is None:
        raise gr.Error('请选择基模型!')
    if user_model is None:
        raise gr.Error('请选择人物lora!')
    if lora_choice is None:
        raise gr.Error('请选择LoRa文件!')
    if style_model is None and lora_choice == 'preset':
        raise gr.Error('请选择风格!')

    base_model = base_models[base_model_index]['model_id']
    matched = list(filter(lambda item: style_model == item['name'], styles))
    style_model = matched[0]['name']

    save_dir = join_worker_data_dir(uuid, 'inference_result', base_model, user_model)
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model)
    else:
        save_dir = os.path.join(save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])

    if not os.path.exists(save_dir):
        return [], []

    if deal_type == "load" or deal_type == "update":
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
    elif deal_type == "delete":
        shutil.rmtree(save_dir)
        return [], []


import os


def load_all_history(uuid, base_model_index=None, user_model=None, lora_choice=None, style_model=None):
    all_single_imgs = []
    all_concat_imgs = []

    base_dir = join_worker_data_dir(uuid, 'inference_result')
    if not os.path.exists(base_dir):
        return [], []

    base_models_to_check = [bm['model_id'].split('/')[0] for bm in base_models] if base_model_index is None else [
        base_models[base_model_index]['model_id'].split('/')[0]]
    styles_to_check = [style['name'] for style in styles] if style_model is None else [style_model]

    for base_model in os.listdir(base_dir):
        if base_model not in base_models_to_check:
            continue
        base_model_dir = os.path.join(base_dir, base_model)
        for user_model_name in os.listdir(base_model_dir):
            if user_model is not None and user_model_name != user_model:
                continue
            user_model_dir = os.path.join(base_model_dir, user_model_name)
            for lora_or_style in os.listdir(user_model_dir):
                if lora_choice is not None and lora_choice != lora_or_style:
                    continue
                model_dir = os.path.join(user_model_dir, lora_or_style)
                for style in styles_to_check:
                    style_dir = os.path.join(model_dir, f"style_{style}")

                    single_dir = os.path.join(style_dir, 'single')
                    concat_dir = os.path.join(style_dir, 'concat')
                    if os.path.exists(single_dir):
                        single_imgs = sorted(os.listdir(single_dir))
                        single_imgs = [(os.path.join(single_dir, img), f"{lora_or_style}&{style}",
                                        os.path.getmtime(os.path.join(single_dir, img))) for img in single_imgs]
                        all_single_imgs.extend(single_imgs)

                    if os.path.exists(concat_dir):
                        concat_imgs = sorted(os.listdir(concat_dir))
                        concat_imgs = [(os.path.join(concat_dir, img), f"{lora_or_style}&{style}",
                                        os.path.getmtime(os.path.join(concat_dir, img))) for img in concat_imgs]
                        all_concat_imgs.extend(concat_imgs)

    # Sort by last modified time in descending order
    all_single_imgs = sorted(all_single_imgs, key=lambda x: x[2], reverse=True)
    all_concat_imgs = sorted(all_concat_imgs, key=lambda x: x[2], reverse=True)

    # Remove the timestamp before returning
    all_single_imgs = [(img, label) for img, label, _ in all_single_imgs]
    all_concat_imgs = [(img, label) for img, label, _ in all_concat_imgs]

    return all_single_imgs, all_concat_imgs


def deal_history(uuid, base_model_index=None, user_model=None, lora_choice=None, style_model=None, deal_type="load"):
    single_imgs, concat_imgs = deal_history_inner(uuid, base_model_index, user_model, lora_choice, style_model,
                                                  deal_type)
    return gr.Gallery(value=single_imgs, visible=True), gr.Gallery(value=concat_imgs, visible=True)


def update_prompt(style_model):
    matched = list(filter(lambda item: style_model == item['name'], styles))
    style = matched[0]
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])
    multiplier_style = style['multiplier_style']
    multiplier_human = style['multiplier_human']
    return gr.Textbox(value=pos_prompt), \
        gr.Slider(value=multiplier_style), \
        gr.Slider(value=multiplier_human)


def launch_pipeline(state,
                    uuid,
                    pos_prompt,
                    neg_prompt=None,
                    base_model_index=None,
                    user_model=None,
                    num_images=1,
                    lora_choice=None,
                    style_model=None,
                    multiplier_style=0.35,
                    multiplier_human=0.95,
                    pose_model=None,
                    pose_image=None,
                    sr_img_size=None,
                    cartoon_style_idx=None,
                    use_lcm_idx=False
                    ):
    global IS_TRAINING
    print(f"IS_TRAINING {IS_TRAINING}")
    if IS_TRAINING:
        raise "当前服务器正有训练任务，请稍后再试"

    try:
        IS_TRAINING = True
        print(f"IS_TRAINING_SET {IS_TRAINING}")
        # time.sleep(120)
        infer_choose_block = gr.Group(visible=False)
        inference_result_block = gr.Group(visible=True)
        state['page'] = "inference_result"

        # # Check base model
        if base_model_index == None:
            raise gr.Error('请选择基模型!')
        set_spawn_method()
        # Check character LoRA
        tmp_character_model = base_models[base_model_index]['model_id']
        if tmp_character_model != 'AI-ModelScope/stable-diffusion-xl-base-1.0':
            tmp_character_model = character_model
        # tmp_character_model = 'AI-ModelScope/stable-diffusion-xl-base-1.0'

        folder_path = join_worker_data_dir(uuid, tmp_character_model)
        folder_list = []
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isdir(file_path):
                    file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                    file_lora_path_swift = f"{file_path}/swift"
                    if os.path.exists(file_lora_path) or os.path.exists(file_lora_path_swift):
                        folder_list.append(file)
        if len(folder_list) == 0:
            raise gr.Error('没有模型分身，请先上传照片训练!')

        # Check output model
        if user_model == None:
            raise gr.Error('请选择模型分身！')
        # Check lora choice
        if lora_choice == None:
            raise gr.Error('请选择模型分身!')
        # Check style model
        if style_model == None and lora_choice == 'preset':
            raise gr.Error('请选择写真风格!')

        base_model = base_models[base_model_index]['model_id']
        revision = base_models[base_model_index]['revision']
        sub_path = base_models[base_model_index]['sub_path']

        before_queue_size = 0
        before_done_count = inference_done_count
        matched = list(filter(lambda item: style_model == item['name'], styles))
        if len(matched) == 0:
            raise gr.Error('写真风格未选择!')
        matched = matched[0]
        style_model = matched['name']

        if lora_choice == 'preset':
            if matched['model_id'] is None:
                style_model_path = None
            else:
                model_dir = snapshot_download(matched['model_id'], revision=matched['revision'])
                style_model_path = os.path.join(model_dir, matched['bin_file'])
        else:
            print(f'uuid: {uuid}')
            temp_lora_dir = join_worker_data_dir(uuid, 'temp_lora')
            file_name = lora_choice
            print(lora_choice.split('.')[-1], os.path.join(temp_lora_dir, file_name))
            if lora_choice.split('.')[-1] != 'safetensors' or not os.path.exists(
                    os.path.join(temp_lora_dir, file_name)):
                raise ValueError(f'Invalid lora file: {lora_file.name}')
            style_model_path = os.path.join(temp_lora_dir, file_name)

        if pose_image is None or pose_model == 0:
            pose_model_path = None
            use_depth_control = False
            pose_image = None
        else:
            model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
            pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')
            if pose_model == 1:
                use_depth_control = True
            else:
                use_depth_control = False

        print("-------user_model: ", user_model)

        use_main_model = True
        use_face_swap = True
        use_post_process = True
        use_stylization = False

        instance_data_dir = join_worker_data_dir(uuid, 'training_data', tmp_character_model, user_model)
        lora_model_path = join_worker_data_dir(uuid, tmp_character_model, user_model)

        gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt,
                                   style_model_path,
                                   multiplier_style, multiplier_human, use_main_model,
                                   use_face_swap, use_post_process,
                                   use_stylization)

        num_images = min(6, num_images)

        with ProcessPoolExecutor(max_workers=5) as executor:
            future = executor.submit(gen_portrait, instance_data_dir,
                                     num_images, base_model, lora_model_path, sub_path, revision, sr_img_size,
                                     cartoon_style_idx, use_lcm_idx=use_lcm_idx)
            while not future.done():
                is_processing = future.running()
                if not is_processing:
                    cur_done_count = inference_done_count
                    to_wait = before_queue_size - (cur_done_count - before_done_count)
                    yield [state, infer_choose_block, inference_result_block,
                           "排队等待资源中, 前方还有{}个生成任务, 预计需要等待{}分钟...".format(to_wait, to_wait * 2.5),
                           None]
                else:
                    yield [state, infer_choose_block, inference_result_block, "正在为你生成中，请耐心等待...", None]
                time.sleep(1)

        outputs = future.result()
        outputs_RGB = []
        for out_tmp in outputs:
            outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

        save_dir = join_worker_data_dir(uuid, 'inference_result', base_model, user_model)
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

            yield [state, infer_choose_block, inference_result_block, "生成完成，可点击喜欢的造型进行查看或保存。",
                   outputs_RGB]
        else:
            yield [state, infer_choose_block, inference_result_block, "生成失败, 请重试!", outputs_RGB]
    finally:
        IS_TRAINING = False


def inference_other(state):
    infer_choose_block = gr.Group(visible=True)
    inference_result_block = gr.Group(visible=False)
    state['page'] = "inference"
    return state, infer_choose_block, inference_result_block


def select_user_model_function(evt: gr.SelectData):
    print("Selected user_model image data:", evt.value)
    name = evt.value['caption']
    return gr.Text(value=name, visible=True)


def select_function(evt: gr.SelectData):
    print("Selected image data:", evt.value)
    name = evt.value['caption']
    return gr.Text(value=name, visible=True)


def dblclick_pic_function(run_button, upload_button, double_click_index, current_images, hint):
    print("double_click_index:", double_click_index)
    if current_images is None:
        return

    if double_click_index is not None and double_click_index.isdigit():
        double_click_index = int(double_click_index)
        if double_click_index < len(current_images):
            current_images.pop(double_click_index)  # 删除双击的图片

    print("new_images:", current_images)
    # 更新界面状态
    if len(current_images) >= 3:
        return gr.Button(visible=True), gr.UploadButton(visible=False), current_images, gr.Markdown(value=hint)
    else:
        global init_hint_content
        return gr.Button(visible=False), gr.UploadButton(visible=True), current_images, gr.Markdown(
            value=init_hint_content)


def upload_file(files, current_images, hint):
    print(f"files: {files}")
    current_images = current_images or []
    if len(current_images) + len(files) >= 3:
        global to_train_hint_content
        new_hint = to_train_hint_content
        return gr.Button(visible=True), gr.Button(visible=False), current_images + files, gr.Markdown(value=new_hint)
    else:
        return gr.Button(visible=False), gr.Button(visible=True), current_images + files, gr.Markdown(value=hint)


def show_train_interface(state):
    """显示训练模型的界面并隐藏主界面。"""
    state['page'] = "train"
    main_content = gr.Column(visible=False)
    train_content = gr.Column(visible=True)
    infer_content = gr.Column(visible=False)
    history_content = gr.Column(visible=False)
    inference_button1 = gr.Button(visible=False)
    global init_hint_content
    return state, main_content, train_content, infer_content, history_content, "", [], inference_button1, init_hint_content


def show_history_interface(state):
    state['page'] = "history"
    main_content = gr.Column(visible=False)
    train_content = gr.Column(visible=False)
    infer_content = gr.Column(visible=False)
    history_content = gr.Column(visible=True)
    inference_button1 = gr.Button(visible=False)

    single_imgs, concat_imgs = deal_history_inner(None)
    global init_hint_content
    return state, main_content, train_content, infer_content, history_content, gr.Gallery(value=single_imgs,
                                                                                          visible=True)


def show_inference_interface(state):
    """显示推理写真的界面并隐藏其他界面。"""
    state['page'] = "inference"

    main_content = gr.Column(visible=False)
    train_content = gr.Column(visible=False)
    infer_content = gr.Column(visible=True)
    history_content = gr.Column(visible=False)
    return state, main_content, train_content, infer_content, history_content


def show_main_interface(state):
    """显示主界面并隐藏训练模型的界面。"""
    state['page'] = "main"
    main_content = gr.Column(visible=True)
    train_content = gr.Column(visible=False)
    infer_content = gr.Column(visible=False)
    history_content = gr.Column(visible=False)
    return state, main_content, train_content, infer_content, history_content


# 用于radio样式的模型名列表
def get_user_model_list():
    uuid = 'qw'
    folder_list = []
    for idx, tmp_character_model in enumerate(['AI-ModelScope/stable-diffusion-xl-base-1.0', character_model]):
        folder_path = join_worker_data_dir(uuid, tmp_character_model)
        if not os.path.exists(folder_path):
            continue
        else:
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isdir(folder_path):
                    file_lora_path = f"{file_path}/pytorch_lora_weights.bin"
                    file_lora_path_swift = f"{file_path}/swift"
                    if os.path.exists(file_lora_path) or os.path.exists(file_lora_path_swift):
                        folder_list.append(file)

    return folder_list


# 用于gallery样式的模型首张图列表
import os


def update_output_model():
    uuid = 'qw'
    base_path = '/root/autodl-tmp/facechain/worker_data/qw/training_data/ly261666/cv_portrait_model'
    model_folders = get_user_model_list()

    image_paths = []
    for model_name in model_folders:
        model_folder_path = os.path.join(base_path, model_name)
        first_image_path = os.path.join(model_folder_path, '000.jpg')
        if os.path.exists(first_image_path):
            # 获取模型文件夹的最后修改时间
            model_creation_time = os.path.getmtime(model_folder_path)
            image_paths.append((first_image_path, model_name, model_creation_time))

    # 对列表按照创建时间倒序排列
    image_paths.sort(key=lambda x: x[2], reverse=True)

    # 删除时间戳信息，仅返回图片路径和模型名
    image_paths = [(path, name) for path, name, _ in image_paths]

    return image_paths


styles = []
for base_model in base_models:
    style_in_base = []
    folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if data['img'][:2] == './':
                data['img'] = f"{project_dir}/{data['img'][2:]}"
            style_in_base.append(data['name'])
            styles.append(data)
    base_model['style_list'] = style_in_base

gallery_double_click_js = """
<script>
    // 使用 document.body 来确保事件监听始终有效
    console.log("打印测试");
    document.body.addEventListener('dblclick', function(event) {
        console.log("检测到双击");
        const clickedElement = event.target;
        // 检查双击的是否是图片，并且这个图片是否位于 .thumbnail-item 类的元素内
        if (clickedElement.tagName === 'IMG' && clickedElement.closest('.thumbnail-item') && clickedElement.closest('.grid-container')) {
            console.log("检测到图片");
            // 选取所有包含图片的按钮元素
            const images = Array.from(document.querySelectorAll('.thumbnail-item img'));
            console.log("images:", images);
            // 获取双击的图片索引
            const index = images.indexOf(clickedElement);
            console.log("双击图片的索引:", index);

            // 获取textarea元素
            const textarea = document.querySelector('#double_click_index textarea');
            // 更新值
            textarea.value = index.toString();
            // 创建并触发input事件，以便Gradio后端能捕捉到这个变化
            const event = new Event('input', { bubbles: true });
            textarea.dispatchEvent(event);

            // 打印赋值后的结果，以确认赋值成功
            console.log("double_click_index: ", textarea.value);

            // 触发后端处理的按钮点击事件
            document.getElementById('instance_images_click_button').click();
            }
        }
    );
          // 获取具有特定类名的元素  
            var element = document.querySelector('.svelte-1rjryqp');  
            // 隐藏元素  
            if (element) {  
                // 或者移除元素  
                 element.parentNode.removeChild(document.querySelector('.svelte-1rjryqp'));  
            }
</script>
"""

cus_theme = gr.Theme.load("themes/theme_land/themes/theme_schema@0.0.3.json")
test_css = """
.show-api.svelte-1rjryqp.svelte-1rjryqp.svelte-1rjryqp {
    display: none !important;  

}
footer.svelte-1rjryqp.svelte-1rjryqp.svelte-1rjryqp {
    display: none !important;  
}
.bginput{
    color: black; 
    font-weight: bold; /* 字体加粗 */  
    font-size: 26px; /* 字体大小，您可以根据需要调整这个值 */   
    background-color: white;

}
#qktp{
  background-color: rgb(213, 210, 210);  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;
}
#xzzpsc{
  background-color: rgb(15, 99, 168); ;  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;
}
#xzzpsc12{
  background-color: rgb(15, 99, 168); ;  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;
}
.scaixz{
  background-color: rgb(15, 99, 168); ;  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;
}
.hdsy{
  background-color: #689aea;  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;
}
#gallery{
    background-color: white;
}
#gallery2{
    background-color: white;
}

.prose * {  
    color: #a59f9b;  
}  
div.empty.svelte-1oiin9d.large.unpadded_box{
    background-color: white;
}
.gradio-container { background-color: white;  }
.button{  
  background-color: rgb(15, 99, 168);  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;}
.lsbjanniuty{  
  background-color: rgb(15, 99, 168);  
    color: white;  
    border: none; 
     border-radius: 11px; 
    padding: 10px 20px;}
.yellow{  
  background-color: #a3a31d;  
    color: white;  
    border: none;  
     border-radius: 11px;
    padding: 10px 20px;}
#lsxzzp{  
  background-color: rgb(255, 163, 52);  
    color: white;  
    border: none;  
    padding: 10px 20px;
     border-radius: 11px;
    }
#step1{  
justify-content: center;
    color: black; 
    font-weight: bold; /* 字体加粗 */  
    font-size: 26px; /* 字体大小，您可以根据需要调整这个值 */   
    }
#step2{  
justify-content: center;
    color: black; 
    font-weight: bold; /* 字体加粗 */  
    font-size: 26px; /* 字体大小，您可以根据需要调整这个值 */   
    }
#input1{
 border-top: none; 
    border: none; /* 去除边框 */  
    background-color: white; /* 如果需要透明背景 */  
    outline: none; /* 去除点击时的蓝色边框（outline） */  
    box-shadow: none; /* 去除阴影 */  
    /* 可以添加其他自定义样式，如字体、颜色等 */  
}
.padded.hide-container{
    background-color: white; /* 如果需要透明背景 */  
}
.logoright{
    justify-content: center; /* 水平居中 */  
        border: none; /* 去除边框 */  

}
    
"""
with gr.Blocks(theme=cus_theme, head=gallery_double_click_js, css=test_css) as demo:
    state = gr.State({"page": "main"})  # 初始状态设置为主界面

    main_content = gr.Column()
    with main_content:
        gr.Markdown("<center><h1 style='color:blue;'>AI写真</h1></center>")
        gr.Image("style_image/shouye.png")
        gr.Markdown("<div class='container'>")
        # gr.Markdown("<center><p>0302期AI产品班实战作品</p></center>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<center><p style='justify-content: center;color: black; font-weight: bold; font-size: 26px; '>step 1:</p></center>",
                    elem_id="step1")
            with gr.Column(scale=3):
                button_train = gr.Button("训练模型制作分身", elem_classes="button")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "<center><p style='justify-content: center;color: black; font-weight: bold; font-size: 26px; '>step 2:</p></center>",
                    elem_id="step1")
            with gr.Column(scale=2):
                button_infer = gr.Button("制作AI写真", elem_classes="button")
        with gr.Row():
            with gr.Column():
                button_history = gr.Button("历史写真作品", elem_classes="button yellow", elem_id="lsxzzp")
        with gr.Row():
            with gr.Column():
                gr.Image("style_image/logo.png",elem_classes='logoright' )
        # with gr.Row():
        #     with gr.Column():
        #         gr.Markdown("<center><p>0302期AI产品班实战作品</p></center>")

        gr.Markdown("</div>")

    train_content = gr.Column()
    train_content.visible = False

    infer_content = gr.Column()
    infer_content.visible = False

    history_content = gr.Column()
    history_content.visible = False

    with infer_content:
        gr.Markdown("<center><h1 style='color:blue;'>AI写真</h1></center>")
        uuid = gr.Text(label="modelscope_uuid", visible=False, value='qw')

        with gr.Column() as infer_choose_block:
            user_model_list = update_output_model()
            # user_model = gr.Radio(label="1. 选择模型分身", choices=user_model_list, type="value")
            with gr.Group(elem_classes='bginput'):
                gr.Markdown("<center><h3 style='background-color: white;'>选择模型分身</h3></center>")

                user_model = gr.Text(label="",interactive=False , elem_classes='bginput')
                user_model_gallery = gr.Gallery(value=update_output_model,
                                                allow_preview=False,
                                                interactive=False,
                                                elem_id="gallery2",

                                                label="分身")

            with gr.Group(elem_classes='bginput'):
                gr.Markdown("<center><h3 style='background-color: white;'>选择写真风格</h3></center>")

                style_model = gr.Text( label="",interactive=False,elem_classes='bginput')
                gallery = gr.Gallery(value=[(item["img"], item["name"]) for item in styles],
                                     label="风格",
                                     allow_preview=False,
                                     interactive=False,
                                     elem_id="gallery",
                                     show_share_button=False)
            inference_button2 = gr.Button('生成AI写真',elem_classes='scaixz')

        with gr.Column() as inference_result_block:
            inference_result_block.visible = False
            with gr.Group():
                infer_progress = gr.Markdown("当前无生成任务")
            with gr.Group():
                output_images = gr.Gallery(label='Output', show_label=False, columns=3, rows=2, height=600,
                                           object_fit="contain")

            inference_again_button = gr.Button('再次生成',elem_classes='lsbjanniuty')
            button_infer_other = gr.Button(value="更换其他风格" ,elem_classes='lsbjanniuty')
        gr.Markdown("<hr>")
        button_home2 = gr.Button("回到首页",elem_classes='hdsy')

        base_model_list = []
        for base_model in base_models:
            base_model_list.append(BASE_MODEL_MAP[base_model['name']])
        print(f"chuckeytang base_model_list {base_model_list}")
        base_model_index = gr.Radio(label="基模型选择", choices=base_model_list, type="index", value="写真模型",
                                    visible=False)
        pos_prompt = gr.Textbox(label="提示语", lines=3,
                                value=generate_pos_prompt(None, styles[0]['add_prompt_style']),
                                interactive=True, visible=False)
        neg_prompt = gr.Textbox(label="负向提示语", lines=3,
                                value="",
                                interactive=True, visible=False)

        lora_file = gr.File(
            value=None,
            label="上传LoRA文件(Upload LoRA file)",
            type="filepath",
            file_types=[".safetensors"],
            file_count="single",
            visible=False,
        )

        if neg_prompt.value == '':
            neg_prompt.value = neg

        num_images = gr.Number(label='生成图片数量', value=3, precision=1, minimum=1, maximum=6, visible=False)
        lora_choice = gr.Dropdown(choices=["preset"], type="value", value="preset", label="LoRA文件(LoRA file)",
                                  visible=False)
        multiplier_style = gr.Slider(minimum=0, maximum=1, value=0.25,
                                     step=0.05, label='风格权重(Multiplier style)', visible=False)
        multiplier_human = gr.Slider(minimum=0, maximum=1.2, value=0.95,
                                     step=0.05, label='形象权重(Multiplier human)', visible=False)

        style_model.change(update_prompt, style_model, [pos_prompt, multiplier_style, multiplier_human], queue=False)

        pmodels = []
        for pmodel in pose_models:
            pmodels.append(pmodel['name'])
        with gr.Accordion("姿态控制(Pose control)", open=False, visible=False):
            with gr.Row():
                pose_image = gr.Image(type='filepath', label='姿态图片(Pose image)', height=250)
                pose_res_image = gr.Image(interactive=False, label='姿态结果(Pose result)', visible=False, height=250)
            gr.Examples(pose_examples['man'], inputs=[pose_image], label='男性姿态示例')
            gr.Examples(pose_examples['woman'], inputs=[pose_image], label='女性姿态示例')
            pose_model = gr.Radio(choices=pmodels, value=pose_models[0]['name'],
                                  type="index", label="姿态控制模型")

        sr_img_size = gr.Radio(label="输出分辨率选择", choices=["512x512"], type="index", value="512x512",
                               visible=False)
        cartoon_style_idx = gr.Radio(label="动漫风格选择", choices=['无', '2D人像卡通', '3D人像卡通化'], type="index",
                                     value="无", visible=False)
        use_lcm_idx = gr.Radio(label="是否使用LCM采样器", choices=['使用默认采样器', '使用LCM采样器'], type="index",
                               value="使用默认采样器", visible=False)

        user_model_gallery.select(select_user_model_function, None, user_model, queue=False)
        gallery.select(select_function, None, style_model, queue=False)
        inference_button2.click(fn=launch_pipeline,
                                inputs=[state, uuid, pos_prompt, neg_prompt, base_model_index, user_model, num_images,
                                        lora_choice, style_model, multiplier_style, multiplier_human,
                                        pose_model, pose_image, sr_img_size, cartoon_style_idx, use_lcm_idx],
                                outputs=[state, infer_choose_block, inference_result_block, infer_progress,
                                         output_images],
                                trigger_mode="once")

        inference_again_button.click(fn=launch_pipeline,
                                     inputs=[state, uuid, pos_prompt, neg_prompt, base_model_index, user_model,
                                             num_images, lora_choice, style_model, multiplier_style, multiplier_human,
                                             pose_model, pose_image, sr_img_size, cartoon_style_idx, use_lcm_idx],
                                     outputs=[state, infer_choose_block, inference_result_block, infer_progress,
                                              output_images],
                                     trigger_mode="once")

        button_infer_other.click(
            inference_other,
            inputs=[state],
            outputs=[state, infer_choose_block, inference_result_block]
        )

    with train_content:
        trainer = Trainer()
        uuid = gr.Text(label="modelscope_uuid", visible=False, value='qw')
        gr.Markdown("<center><h1 style='color:blue;'>AI写真</h1></center>")
        gr.Markdown("<center><h2 style='color:black;'>请输入模型分身名称</h2></center>")
        output_model_name = gr.Textbox(
            placeholder="输入：您的昵称"
            , lines=1,elem_id="input1", container=False
        )
        base_model_name = gr.Dropdown(choices=['AI-ModelScope/stable-diffusion-v1-5',
                                               SDXL_BASE_MODEL_ID],
                                      value='AI-ModelScope/stable-diffusion-v1-5',
                                      label='基模型', visible=False)
        gr.Markdown("<center><h2 style='color:black;'>照片库</h2></center>")

        instance_images = gr.Gallery(
            allow_preview=False, interactive=False
       )
        hint = gr.Markdown(init_hint_content,elem_classes='qiansewenbw')

        # 创建一个UploadButton组件，允许用户上传多个图片文件

        # upload_button = gr.inputs.UploadButton(
        #     label="选择照片上传",
        #     type="files",  # 使用type="files"允许多文件上传
        #     accept=".png, .jpg, .jpeg",  # 限制文件类型为图片
        #     multiple=True,  # 允许选择多个文件
        #     # 注意：Gradio本身不提供双击移除图片的功能
        # )

        upload_button = gr.UploadButton("选择照片上传", file_types=["image"], file_count="multiple",elem_id='xzzpsc')
        run_button = gr.Button('开始训练模型分身', visible=False, elem_id='xzzpsc12')
        clear_button = gr.Button("清空已上传图片" ,elem_id='qktp')
        gr.Markdown("<hr>")
        button_home1 = gr.Button("回到首页",elem_classes='hdsy')
        inference_button1 = gr.Button('生成AI写真', visible=False,elem_classes='scaixz')

        double_click_index = gr.Textbox(visible=False, elem_id="double_click_index")  # 用于接收双击的图片索引
        instance_images_click_button = gr.Button("处理图片", elem_id="instance_images_click_button",
                                                 visible=False)  # 模拟gallery中图片处理按钮


        # 绑定选择事件
        instance_images_click_button.click(dblclick_pic_function,
                                           inputs=[run_button, upload_button, double_click_index, instance_images,
                                                   hint], outputs=[run_button, upload_button, instance_images, hint])

        upload_button.upload(upload_file, inputs=[upload_button, instance_images, hint],
                             outputs=[run_button, upload_button, instance_images, hint], queue=False)
        clear_button.click(
            fn=lambda: [gr.Button(visible=False), gr.UploadButton(visible=True), None, gr.Markdown(value=hint.value)],
            inputs=None, outputs=[run_button, upload_button, instance_images, hint])

        run_button.click(fn=trainer.run,
                         inputs=[
                             uuid,
                             base_model_name,
                             instance_images,
                             output_model_name,
                         ],
                         outputs=[run_button, upload_button, clear_button, inference_button1, hint, user_model_gallery])

    with history_content:
        history_images = gr.Gallery(label="历史写真", allow_preview=True, interactive=False, columns=3)
        gr.Markdown("<hr>")
        button_home3 = gr.Button("回到首页",elem_classes='hdsy')

    button_train.click(show_train_interface, inputs=[state],
                       outputs=[state, main_content, train_content, infer_content, history_content,
                                output_model_name, instance_images, inference_button1, hint])
    button_infer.click(show_inference_interface, inputs=[state],
                       outputs=[state, main_content, train_content, infer_content, history_content])
    inference_button1.click(show_inference_interface, inputs=[state],
                            outputs=[state, main_content, train_content, infer_content, history_content])
    button_home1.click(show_main_interface, inputs=[state],
                       outputs=[state, main_content, train_content, infer_content, history_content])
    button_home2.click(show_main_interface, inputs=[state],
                       outputs=[state, main_content, train_content, infer_content, history_content])
    button_home3.click(show_main_interface, inputs=[state],
                       outputs=[state, main_content, train_content, infer_content, history_content])
    button_history.click(show_history_interface, inputs=[state],
                         outputs=[state, main_content, train_content, infer_content, history_content, history_images])

if __name__ == "__main__":
    set_spawn_method()
    if os.path.exists("/.dockerenv"):
        demo.queue(status_update_rate=1).launch(server_name="0.0.0.0", share=True, server_port=6006)
    else:
        demo.queue(status_update_rate=1).launch(share=True, server_port=6006)
