import torch, argparse, os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from voxa.speakermgr.speaker_manager import SpeakerManager

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data', type=str, default='vctk', help='use specific dataset')
    parser.add_argument('--init_from', type=str, default='', help='load parameters from...')
    parser.add_argument('--class_type', type=str, default='gender', help='gender, age, accent, region')
    parser.add_argument('--load_spkr', type=int, default=0, help='load speaker embedding layer')
    parser.add_argument('--exp_no', type=str, default='', help='exp_no')
    args = parser.parse_args()

    save_dir = 'visualization/' + args.exp_no + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # print(valid_idx)
    if args.load_spkr:
        sm = SpeakerManager(args.data.split(','))
        # load from a saved model's speaker embedding layer
        checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
        old_data_list = checkpoint['args'].data.split(',')
        old_embedding_matrix = checkpoint['state_dict']['spkr_embed.weight']
        embedding = sm.get_adapted_speaker_embedding_matrix(old_data_list, old_embedding_matrix)
        embedding = embedding.cpu().numpy()
    else:
        checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
        sm = SpeakerManager(checkpoint['args'].data.split(','))
        embedding = checkpoint['state_dict']['spkr_embed.weight']

    info = {}
    set_gender, set_age, set_accent = set(), set(), set()
    with open('util/speaker-info.txt', 'r') as rFile:
        line = rFile.readline()     # skip index row
        line = rFile.readline()
        while line:
            tmp = line.strip().split(' ')
            sp = [item for item in tmp if item != ""]

            if len(sp) > 4:
                sp[4] = " ".join(sp[4:])
            elif len(sp) < 4:
                sp[4] = sp[3]

            set_age.add(int(sp[1])//10*10)
            set_gender.add(sp[2])
            set_accent.add(sp[3])

            info[sm.get_compact_id_from_joined(sp[0])] = sp[1:]
            line = rFile.readline()

    print('# age/gender/accent: %d, %d, %d' % (len(set_age), len(set_gender), len(set_accent)))
    color_table = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    class_id = {}
    if args.class_type == 'age':
        label_table = [10, 20, 30]
        for i in sorted(info.keys()):
            j = info[i]

            if int(j[0])//10*10 == 10:
                class_id[i] = 0
            elif int(j[0])//10*10 == 20:
                class_id[i] = 1
            elif int(j[0])//10*10 == 30:
                class_id[i] = 2
    elif args.class_type == 'gender':
        label_table = ['M', 'F']
        for i in sorted(info.keys()):
            j = info[i]

            if j[1] == 'M':
                class_id[i] = 0
            else:
                class_id[i] = 1
    elif args.class_type == 'accent':
        label_table = ['GB', 'IRE', 'ASIA', 'AM', 'SH']

        region_gb = set(['English', 'Welsh', 'Scottish'])
        region_ire = set(['Irish',  'NorthernIrish'])
        region_asia = set(['Indian' ])
        region_am = set(['Canadian','American'])
        region_sh = set(['NewZealand', 'Australian','SouthAfrican'])

        for i in sorted(info.keys()):
            j = info[i]

            if j[2] in region_gb:
                class_id[i] = 0
            elif j[2] in region_ire:
                class_id[i] = 1
            elif j[2] in region_asia:
                class_id[i] = 2
            elif j[2] in region_am:
                class_id[i] = 3
            elif j[2] in region_sh:
                class_id[i] = 4
            else:
                class_id[i] = 5

    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # legend_label = set([])
    # for i, cls in enumerate(class_id):
    #     if label_table[cls] in legend_label:
    #         current_label = None
    #     else:
    #         current_label = label_table[cls]
    #         legend_label.add(label_table[cls])
    #     ax.scatter(embedding[i, 0], embedding[i, 1], c=color_table[cls], label=current_label)
    # ax.legend()
    # plt.savefig(save_dir + 'raw.png')
    # plt.close('all')
    #
    #
    # tsne = TSNE(n_components=2,n_iter=100000, learning_rate=200).fit_transform(embedding)
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # legend_label = set([])
    # for i, cls in enumerate(class_id):
    #     if label_table[cls] in legend_label:
    #         current_label = None
    #     else:
    #         current_label = label_table[cls]
    #         legend_label.add(label_table[cls])
    #     ax.scatter(tsne[i, 0], tsne[i, 1], c=color_table[cls], label=current_label)
    # ax.legend()
    # plt.savefig(save_dir + 'tsne.png')
    # plt.close('all')

    import numpy as np

    font = {'size'   : 12}
    # font = {'size'   : 22}
    plt.rc('font', **font)



    tsne = TSNE(n_components=2, n_iter=100000, learning_rate=200).fit_transform(embedding)

    pca = PCA(n_components=2, whiten=True).fit(embedding)
    embedding_pca = pca.transform(embedding)

    # plot tsne
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    legend_label = set([])
    for i, p_number in enumerate(sorted(sm.get_all_compact_id(single=False))):
        if label_table[class_id[p_number]] in legend_label:
            current_label = None
        else:
            current_label = label_table[class_id[p_number]]
            legend_label.add(label_table[class_id[p_number]])
        ax.scatter(tsne[i, 0], tsne[i, 1], c=color_table[class_id[p_number]], label=current_label)

        # if class_id[p_number] == 1:
        #     matplotlib.pyplot.text(tsne[i, 0], tsne[i, 1], p_number[-3:], fontsize=7, withdash=False)

        data, id = sm.get_original_id(p_number)
        if id == 0:
            name = data
        else:
            name = id
        matplotlib.pyplot.text(tsne[i, 0], tsne[i, 1], name, fontsize=6, withdash=False)
    ax.legend()
    plt.savefig(save_dir + 'tsne.png')

    # plot pca
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    legend_label = set([])
    for i, p_number in enumerate(sorted(sm.get_all_compact_id(single=False))):
        if label_table[class_id[p_number]] in legend_label:
            current_label = None
        else:
            current_label = label_table[class_id[p_number]]
            legend_label.add(label_table[class_id[p_number]])
        ax.scatter(embedding_pca[i, 0], embedding_pca[i, 1], c=color_table[class_id[p_number]], label=current_label)
#         ax.scatter(embedding_pca[i, 3], embedding_pca[i, 1], c=color_table[cls], label=current_label)
#         if class_id[p_number] == 1:
#             matplotlib.pyplot.text(embedding_pca[i, 0], embedding_pca[i, 1], p_number[-3:], fontsize=7, withdash=False)

        data, id = sm.get_original_id(p_number)
        if id == '0':
            name = data
        else:
            name = id
        matplotlib.pyplot.text(embedding_pca[i, 0], embedding_pca[i, 1], name, fontsize=6, withdash=False)

    ax.legend()
    plt.savefig(save_dir + 'pca.png')
    plt.close('all')


if __name__ == '__main__':
    main()
