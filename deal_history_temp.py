def deal_history_inner(uuid, base_model_index=None , user_model=None, lora_choice=None, style_model=None, deal_type="load"):
    if not uuid:
        if os.getenv("MODELSCOPE_ENVIRONMENT") == 'studio':
            raise gr.Error("请登陆后使用! (Please login first)")
        else:
            uuid = 'qw'
            
    if deal_type == "update":
        if (base_model_index is None) or (user_model is None) or (lora_choice is None) or (style_model is None and lora_choice == 'preset'):
            return gr.Gallery(value=[], visible=True), gr.Gallery(value=[], visible=True) # error triggered by option change, won't pop up warning
        
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
        return gr.Gallery(value=[], visible=True), gr.Gallery(value=[], visible=True)
    
    if deal_type == "load" or deal_type == "update":
        single_dir = os.path.join(save_dir, 'single')
        concat_dir = os.path.join(save_dir, 'concat')
        single_imgs = []
        concat_imgs = []
        if os.path.exists(single_dir):
            single_imgs = sorted(os.listdir(single_dir))
            single_imgs = [os.path.join(single_dir, img) for img in single_imgs]
        if os.path.exists(concat_dir):
            concat_imgs = sorted(os.listdir(concat_dir))
            concat_imgs = [os.path.join(concat_dir, img) for img in concat_imgs]
        return single_imgs, concat_imgs
    elif deal_type == "delete":
        shutil.rmtree(save_dir)
        return [], []