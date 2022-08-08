

def support_deprecated_model_gst(exp_no, model_version):
    print(exp_no, model_version)
    if model_version == 'cats4':
        import tacotron.model_gst as model_gst_code
    elif model_version == 'cats3':
        import tacotron.module.deprecated_model.model_cats_v3.model_gst as model_gst_code
    elif model_version == 'cats2':
        import tacotron.module.deprecated_model.model_cats_v2.model_gst as model_gst_code
    elif model_version == 'cats2_jpn':
        import tacotron.module.deprecated_model.model_cats_v2_jpn.model_gst as model_gst_code
    elif model_version == 'cats3_fluent':
        import tacotron.module.deprecated_model.model_cats_v3_fluent.model_gst as model_gst_code
    elif model_version == 'cats':
        import tacotron.module.deprecated_model.model_cats_v1.model_gst as model_gst_code
    elif model_version == 'align':
        if exp_no.startswith('2105'):
            import tacotron.module.deprecated_model.model_align_v1.model_gst as model_gst_code
        else:
            import tacotron.module.deprecated_model.model_cats_v1.model_gst as model_gst_code
    elif model_version == '':
        if exp_no in ["200", "312", "314", "325", "335", "351", "356", "359", "360", "361", "384", "434", "471", "485", "yk8"]:
            import tacotron.module.deprecated_model.model_typecast1909 as model_gst_code
        elif exp_no in ['418', '455', '458', '470', '482']:
            import tacotron.module.deprecated_model.model_audiobook1912 as model_gst_code
        elif exp_no.startswith('single_mel_'):
            import tacotron.module.deprecated_model.model_single_mel as model_gst_code
        else:
            import tacotron.module.deprecated_model.model_arattention.model_gst as model_gst_code
    else:
        raise RuntimeError('The model is not supported.')
    return model_gst_code
