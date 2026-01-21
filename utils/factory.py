def get_model(model_name, args):
    name = model_name.lower()
    if name == 'ldeprompt':
        from models.ldeprompt import Learner
    else: 
        assert 0
    return Learner(args)