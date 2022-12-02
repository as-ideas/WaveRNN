import torch
from models.generator import Generator
from models.multiscale import MultiScaleDiscriminator

if __name__ == '__main__':

    model_path = '/Users/cschaefe/workspace/tts-synthv3/app/11111111/models/bild_voice/voc_model/model.pt'
    model_g = Generator(80)
    model_d = MultiScaleDiscriminator()

    optim_g = torch.optim.Adam(model_g.parameters(),
                               lr=0.0001, betas=(0.5, 0.9))
    optim_d = torch.optim.Adam(model_d.parameters(),
                               lr=0.0001, betas=(0.5, 0.9))


    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_g.load_state_dict(checkpoint['model_g'], strict=False)
    model_d.load_state_dict(checkpoint['model_d'])
    optim_g.load_state_dict(checkpoint['optim_g'])
    optim_d.load_state_dict(checkpoint['optim_d'])

    print(model_g)