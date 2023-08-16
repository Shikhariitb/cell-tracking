from unet import *
from dataset import *
from torchinfo import summary
import math
from moviepy.editor import ImageSequenceClip

OUTPUT_CLASSES = 1
BATCH_SIZE = 1
IM_SIZE = 1024
PATH = '../results/model/unet.pt'


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = ((logit.data-target.data).abs() <= 1e-1).sum()
    # print(corrects)
    accuracy = (100.0 * corrects)/(IM_SIZE*IM_SIZE)
    return accuracy.item()


# model = Unet(OUTPUT_CLASSES)
model = torch.load(PATH)
# print(model)
# summary(model, input_size=(1, 1, IM_SIZE, IM_SIZE))

learning_rate = 0.0001
num_epochs = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    frames = []

    model = model.train()

    # training step
    for i, (images, labels) in enumerate(train_generator(1+epoch % 2, BATCH_SIZE), 0):

        images = images.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        logits = model(images)
        frames.append((logits.cpu().detach().numpy()
                       * 200).astype(np.uint16).reshape([IM_SIZE, IM_SIZE, 1]))
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()

        # update model params
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, BATCH_SIZE)
        s = f"image: {i:03}, loss: {loss.detach().item():4f} ["
        for j in range(math.floor((i+1)/4)):
            s += '#'
        for j in range(23-math.floor((i+1)/4)):
            s += '_'
        s += ']'
        print(s, end='\r')

    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f            '
          % (epoch+10, train_running_loss / (i+1), train_acc/(i+1)))
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_gif(
        f'../results/train/epoch{epoch+10:02}.gif', fps=10, verbose=False, logger=None)

    torch.save(model, PATH)
