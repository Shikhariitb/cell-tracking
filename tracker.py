import torch
from dataset import *
from queue import Queue
import math
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from moviepy.editor import ImageSequenceClip

BATCH_SIZE = 1
IM_SIZE = 1024
PATH = '../results/model/unet.pt'


def get_cell_masks(inmask):
    id = 1
    statemask = [[-1 for i in range(IM_SIZE)] for j in range(IM_SIZE)]
    outmasks = []
    for i in range(IM_SIZE):
        for j in range(IM_SIZE):
            if statemask[i][j] != -1:
                continue
            elif inmask[i][j] == 0:
                statemask[i][j] = 0
            else:
                # BFS to find connected components
                q = Queue(maxsize=0)
                q.put((i, j))
                statemask[i][j] = id
                outmask = [[0 for i in range(IM_SIZE)] for j in range(IM_SIZE)]
                outmask[i][j] = 1
                while(q.qsize() != 0):
                    s = q.get()
                    # print(s)
                    for a in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        sa = (s[0]+a[0], s[1]+a[1])
                        if sa[0] >= 0 and sa[1] >= 0 and sa[0] < IM_SIZE and sa[1] < IM_SIZE and (statemask[sa[0]][sa[1]] == -1) and inmask[sa[0]][sa[1]] == 1:
                            q.put(sa)
                            if inmask[sa[0]][sa[1]] == 0:
                                statemask[sa[0]][sa[1]] = 0
                            else:
                                statemask[sa[0]][sa[1]] = id
                                outmask[sa[0]][sa[1]] = 1
                id += 1
                outmasks.append(np.array(outmask))
    return statemask, outmasks


SMOOTH = 1e-4


def get_centroid(mask):
    x = 0
    y = 0
    for i in range(IM_SIZE):
        for j in range(IM_SIZE):
            x += mask[i][j]*i / IM_SIZE
            y += mask[i][j]*j / IM_SIZE
    return [x, y]


def get_capacity(u, v):
    return 2*((np.multiply(u, v).sum() + SMOOTH) / (np.add(u, v).sum() + SMOOTH))


def get_appearance_capacity(u):
    uc = get_centroid(u)
    e = [np.array([0, uc[1]]), np.array([IM_SIZE-1, uc[1]]),
         np.array([uc[0], 0]), np.array([uc[0], IM_SIZE-1])]
    uc = np.array(uc)
    return math.exp(-min([np.linalg.norm(uc-e[i]) for i in range(4)]) / 500)


def get_max_flow(cell_masks1, cell_masks2):
    m = len(cell_masks1)
    n = len(cell_masks2)

    G = nx.DiGraph()

    # source edges
    for i in range(m):
        G.add_edge("s", f"x_{i:02}", capacity=1.0)

    # source to add node
    G.add_edge("s", "a", capacity=50.0)

    # move edges
    for i in range(m):
        for j in range(n):
            G.add_edge(f"x_{i:02}", f"y_{j:02}", capacity=get_capacity(
                cell_masks1[i], cell_masks2[j]))

    # add edges
    for i in range(n):
        G.add_edge("a", f"y_{i:02}",
                   capacity=get_appearance_capacity(cell_masks2[i]))

    # add to delete
    G.add_edge("a", "d", capacity=50.0)

    # delete edges
    for i in range(m):
        G.add_edge(f"x_{i:02}", "d",
                   capacity=get_appearance_capacity(cell_masks1[i]))

    # sink edges
    for i in range(n):
        G.add_edge(f"y_{i:02}", "t", capacity=1.0)

    # delete node to sink
    G.add_edge("d", "t", capacity=50.0)

    return edmonds_karp(G, "s", "t")

trajectories = {}
frames = []
used = 0

model = torch.load(PATH)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for i, (images) in enumerate(test_generator(1, BATCH_SIZE), 0):
        images = images.to(device)
        labels = model(images)
        mask = np.rint(labels.cpu().detach(
        ).numpy().reshape([IM_SIZE, IM_SIZE]))
        orignal_im = images.cpu().detach().numpy().reshape(IM_SIZE, IM_SIZE)
        # print('begin', end='\r')
        if i == 0:
            prev_labels, prev_masks = get_cell_masks(mask)
            # print(len(prev_masks))
            prev_map = {}
            for index, mask in enumerate(prev_masks):
                trajectories[used] = [np.multiply(orignal_im, mask).astype(
                    np.uint8).reshape([IM_SIZE, IM_SIZE, 1])]
                prev_map[index] = used
                used += 1
        else:
            curr_labels, curr_masks = get_cell_masks(mask)
            # print(len(curr_masks))
            curr_map = {}
            R = get_max_flow(prev_masks, curr_masks)
            for n, nbrsdict in R.adjacency():
                if n.split('_')[0] == 'y':
                    max_index = -1
                    max_flow = 0
                    my_index = int(n.split('_')[1])
                    for nbr, eattr in nbrsdict.items():
                        if nbr.split('_')[0] == 'x' and -1*eattr["flow"] > max_flow:
                            max_index = int(nbr.split('_')[1])
                            max_flow = -1*eattr["flow"]
                        elif nbr == 'a' and -1*eattr["flow"] > max_flow:
                            max_index = -2
                            max_flow = -1*eattr["flow"]
                    if max_index == -2:
                        trajectories[used] = [np.multiply(
                            orignal_im, curr_masks[my_index]).astype(np.uint8).reshape([IM_SIZE, IM_SIZE, 1])]
                        curr_map[my_index] = used
                        used += 1
                    elif max_index >= 0:
                        if max_index != my_index:
                            trajectories[prev_map[max_index]].append(np.multiply(
                            orignal_im, curr_masks[my_index]).astype(np.uint8).reshape([IM_SIZE, IM_SIZE, 1]))
                        curr_map[my_index] = prev_map[max_index]
            prev_labels = curr_labels
            prev_masks = curr_masks
            prev_map = curr_map

        s = f"image: {i:03}, ["
        for j in range(math.floor((i+1)/4)):
            s += '#'
        for j in range(23-math.floor((i+1)/4)):
            s += '_'
        s += ']'
        print(s, end='\r')
        for traj in trajectories.keys():
            # print((traj))
            clip = ImageSequenceClip(trajectories[traj], fps=10)
            clip.write_gif(
                f'../results/track/track{traj}.gif', fps=10, verbose=False, logger=None)
        frames.append(images.numpy().astype(
            np.uint8).reshape([IM_SIZE, IM_SIZE, 1])*5)
clip = ImageSequenceClip(frames, fps=10)
clip.write_gif(
    f'../results/track/input.gif', fps=10, verbose=False, logger=None)
