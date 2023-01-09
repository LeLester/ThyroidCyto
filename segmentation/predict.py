from CA25net import *
import scipy.io
import numpy as np
import time
from PIL import Image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
test_data = XuDataset(os.path.join('dataset', 'test'))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

model = Cia(in_channels=3).to(device)
checkpoint = torch.load('weights/CA25_n.ptf')
model.load_state_dict(checkpoint['model_state_dict'])

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for bt, data in enumerate(test_loader):
    model.train()
    img, _, _, name = data
    img = img.cuda()
    mout, bout = model(img)
    mout = mout.detach().cpu().numpy()[0, 0]
    mout = ((mout > 0.5) * 255).astype('uint8')

    mask = Image.fromarray(mout)
    mask.save('mask_prediction/{}.png'.format(os.path.splitext(name[0])[0]))
