# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import argparse, multiprocessing, torch, yaml, os, sys
import numpy as np

from sklearn import mixture, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
from tts_text_util.get_vocab import get_sg_vocab
from voxa import VoxaConfig
from voxa.database.data_group import expand_data_group

# Add path to use
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_PATH)        # insert at 1, 0 is the script path (or '' in REPL)
from tacotron.loader import DataLoader
from tacotron.util.default_args import load_and_override_args
from tacotron.util.train_common import set_default_GST
skip_spkr_path = os.path.join(ROOT_PATH, 'tacotron/util/style_skip_list.txt')

def plot_cluster(X_sty, cluster_list, outname, original_spkr_id):
    pca = PCA(2) 
    pca_result = pca.fit_transform(np.concatenate([X_sty,cluster_list], axis=0))
    
    pca_all = pca_result[:len(X_sty)] 
    pca_centroid = pca_result[len(X_sty):] 
    plt.clf()

    plt.plot(pca_all[:,0],pca_all[:,1],'bo')
    plt.plot(pca_centroid[:,0],pca_centroid[:,1],'r*')

    # zip joins x and y coordinates in pairs
    for i, (x,y) in enumerate(zip(pca_centroid[:,0],pca_centroid[:,1])):

        label = f"{i}"

        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center') 

    plt.savefig(f'plot/{outname}_{original_spkr_id}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='training script')
    # clustering
    parser.add_argument('--mode', type=str, default='gmm', help='clustering method: gmm, vblda...')
    parser.add_argument('--num_clusters', type=int, default=3, help='number of clusters')
    parser.add_argument('--manual_cluster_from', type=str, default='',
                        help='load list of wav files of each cluster from ...')
    parser.add_argument('--target_spkrs', type=str, default='', help='do clustering only for these speakers')
    # misc
    parser.add_argument('--init_from', type=str, default='', help='load parameters from...')
    parser.add_argument('--out_path', type=str, default='', help='write new check point to ...')
    parser.add_argument('--max_num_style', type=int, default=-1,
                        help='use limited number of style vectors to speed up.')
    parser.add_argument('--set_default_gst', type=int, default=0, help='set 1 to save default gst')
    parser.add_argument('--use_ssd', type=int, default=0, help='set 1 to load from ssd')
    parser.add_argument('--gpu', type=int, nargs='+', help='index of gpu machines to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
    ckpt_args = load_and_override_args(checkpoint['args'], checkpoint['args'])

    # manually set ckpt_args to suitable for prosody extraction
    ckpt_args.shuffle_data = 0  # no shuffling to keep order of audio files same as the metadata
    ckpt_args.batch_size = 1
    ckpt_args.pretraining = 1
    ckpt_args.balanced_spkr = 0
    ckpt_args.trunc_size = ckpt_args.spec_limit

    if ckpt_args.gst == 1:
        from tacotron.model_gst import Tacotron as Tacotron
    elif ckpt_args.gst == 2:
        from tacotron.model_finegrained2 import Tacotron as Tacotron
        
    if args.gpu is None:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu[0])

    vocab, idx_to_vocab = get_sg_vocab()
    model = Tacotron(ckpt_args, vocab=vocab, idx_to_vocab=idx_to_vocab)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.reset_decoder_states(debug=ckpt_args.debug)
    model = model.eval()
    print('loaded checkpoint %s' % (args.init_from))

    if args.use_gpu:
        model = model.cuda()

    voxa_config = VoxaConfig.load_from_config(checkpoint['config'], initialize=True)
    speaker_list = set(ckpt_args.data.split(','))
    if len(args.target_spkrs) > 0:
        target_list = set(expand_data_group(args.target_spkrs.split(','), voxa_config))
    else:
        # Get list of speakers to skip
        skip_list = set([])
        with open(skip_spkr_path, 'r') as rFile:
            line = rFile.readline()
            while line:
                skip_list.add(line.strip())
                line = rFile.readline()

        # exclude speakers in skip_list if not explicitly given in the args.
        target_list = speaker_list - skip_list

    target_list = list(target_list)
    speaker_list = list(speaker_list)

    if args.use_ssd == 1:
        load_from_ssd = True 
    else:
        load_from_ssd = False
    loader = DataLoader(
        ckpt_args,
        target_list,
        voxa_config,
        speaker_list=speaker_list,
        sort=False,
        ssd=load_from_ssd
    )
    num_spkrs = loader.speaker_manager.get_num_speakers()

    if args.set_default_gst == 1:
        print('set GST')
        model = model.train()
        set_default_GST(loader, model, ckpt_args)
        checkpoint['state_dict'] = model.state_dict()
        model = model.eval()

    # print average speech rate
    avg_speed = model.prosody_stats.speed.data
    print('Average speech rate of each speaker')
    for i in range(avg_speed.size(0)):
        print(f'{loader.speaker_manager.get_original_id(i)}: {avg_speed[i].item()}')

    style_vec_list_by_spkr = [[] for _ in range(num_spkrs)]
    # att_weights_list_by_spkr = [[] for _ in range(num_spkrs)]
    print('Extracting style vectors...')
    with torch.no_grad():
        for _ in tqdm(range(loader.split_sizes['whole'])):
            loader_dict = loader.next_batch('whole')
            phoneme_input = loader_dict.get("x_phoneme")
            target_mel = loader_dict.get("y_specM")
            spkr_id = loader_dict.get("idx_speaker")

            if ckpt_args.debug == 10:
                N, T_enc = phoneme_input.size(0), phoneme_input.size(1)
                lang_id = []
                for i in range(N):
                    for t in range(1, T_enc, 1):
                        if t > 0 and t <= 70:
                            lang_id.append(0)
                        elif t > 70 and t <= 127:
                            lang_id.append(1)
                        else:
                            continue
                        break
                assert len(lang_id) == N
                lang_id = torch.LongTensor([lang_id]).type_as(spkr_id).squeeze(0)

                spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S
                lang_vec = model.lang_embed(lang_id).unsqueeze(1)                # N x 1 x S
                spkr_vec = torch.cat([spkr_vec, lang_vec], dim=-1)
            else:
                spkr_vec = model.spkr_embed(spkr_id).unsqueeze(1)                # N x 1 x S

            whole_spec_mask = torch.ones(target_mel.size(1), device=target_mel.device).view(1, -1)
            ref_out_dict = model.ref_encoder(
                target_mel,
                whole_spec_mask,
                spkr_vec,
                debug=ckpt_args.debug
            )
            # ref_out_dict = model.ref_encoder(target_mel, spkr_vec=spkr_vec, debug=ckpt_args.debug)    # old-style ref_encoder

            style_vec = ref_out_dict['gst']
            style_vec = style_vec.squeeze(1)  # 1 x style_dim
            style_vec_list_by_spkr[spkr_id.item()].append(style_vec.cpu().numpy())
            # att_weights = ref_out_dict['att_weights']                                                 # old-style ref_encoder
            # att_weights = att_weights.squeeze(2)  # 1 x n_token                                       # old-style ref_encoder
            # att_weights_list_by_spkr[spkr_id.item()].append(att_weights.cpu().numpy())                # old-style ref_encoder

    if args.max_num_style > 0:
        for i, l in enumerate(style_vec_list_by_spkr):
            max_num = min(len(l), args.max_num_style)
            mask = [np.random.rand() for _ in range(len(l))]
            threshold = np.partition(mask, -max_num)[-max_num]
            mask = [1 if x >= threshold else 0 for x in mask]

            style_vec_list_filtered = []
            # att_weights_list_filtered = []
            for j, m in enumerate(mask):
                if m == 1:
                    style_vec_list_filtered.append(style_vec_list_by_spkr[i][j])
                    # att_weights_list_filtered.append(att_weights_list_by_spkr[i][j])

    # Get cluster centers
    if 'cluster_list' in checkpoint.keys():
        cluster_list = checkpoint['cluster_list']
        if args.out_path == "":
            new_save_path = args.init_from
        else:
            new_save_path = args.out_path
        print('Cluster list already exists. Will be overwritten with new clusters.')
    else:
        cluster_list = [[] for _ in range(num_spkrs)]
        if args.out_path == "":
            new_save_path = args.init_from[:-3] + "sty.t7"
        else:
            new_save_path = args.out_path

    print(f'Clustering: {", ".join(target_list)}')
    if args.manual_cluster_from != '':
        manual_cluster_list_by_spkr = yaml.safe_load(open(args.manual_cluster_from, 'r'))
    else:
        manual_cluster_list_by_spkr = {}

    for spkr_id in tqdm(range(num_spkrs)):
        if not loader.speaker_manager.get_original_id(spkr_id) in target_list:
            continue

        style_vec_list = style_vec_list_by_spkr[spkr_id]
        # att_weights_list = att_weights_list_by_spkr[spkr_id]
        current_cluster_list = cluster_list[spkr_id]

        if len(style_vec_list) > 0:
            # set 0th cluster as the mean style vector
            mean_style = np.mean(style_vec_list, axis=0).squeeze()
            current_cluster_list.append(mean_style)

        original_spkr_id = loader.speaker_manager.get_original_id(spkr_id)
        if original_spkr_id in manual_cluster_list_by_spkr.keys():
            manual_cluster_list = manual_cluster_list_by_spkr[original_spkr_id]
            current_cluster_list.extend(get_cluster_centers_manually(manual_cluster_list, style_vec_list))
        else:
            # Add all gst to style_cluster if dataset is too small
            if args.num_clusters > len(style_vec_list):
                current_cluster_list.extend([s.squeeze() for s in style_vec_list])
                current_cluster_list.extend([mean_style] * (args.num_clusters - len(style_vec_list)))
            else:
                try:
                    current_cluster_list.extend(clustering(args.mode, args.num_clusters, style_vec_list))
                    # plot_cluster(np.concatenate(style_vec_list, axis=0), current_cluster_list, args.mode, original_spkr_id)
                    # current_cluster_list.extend(clustering(args.mode, args.num_clusters, style_vec_list, att_weights_list))
                except ValueError:
                    print(f"ERROR occured while clustering {original_spkr_id}")
                    import pdb;
                    pdb.set_trace()
                     
    # Inject obtained cluster centers to the checkpoint.
    checkpoint['cluster_list'] = cluster_list
    torch.save(checkpoint, new_save_path)
    print(f'Style is indexed for {len(target_list)} speakers. Check: {new_save_path}')


def get_cluster_centers_manually(manual_cluster_list, style_vec_list):
    cluster_list = []
    for ith_cluster in sorted(manual_cluster_list.keys()):
        style_vec_selected = [style_vec_list[j] for j in manual_cluster_list[ith_cluster]]
        cluster_center = np.mean(style_vec_selected, axis=0).squeeze()
        cluster_list.append(cluster_center)
    return cluster_list


# def clustering(mode, n_cluster, style_vec_list, att_weights_list):
def clustering(mode, n_cluster, style_vec_list):
    """
    :param mode: clustering algorithm.
    :param n_cluster: number of clusters
    :param style_vec_list:
    :param att_weights_list:
    :return:
    """
    cluster_list = []
    log_txt = ''

    if mode == 'gmm':
        X = np.concatenate(style_vec_list, axis=0)

        lowest_bic = np.infty
        bic = []
        cv_types = ['diag']
        # cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            n_components = max(min(len(style_vec_list)//50, 20), n_cluster)
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, max_iter=1000)
            gmm.fit(X)
            b = gmm.bic(X)
            log_txt += f"{cv_type}-{n_components} components, bic={b}\n"
            bic.append(b)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        # uncomment this to save all statistics
        # w = torch.from_numpy(best_gmm.weights_)
        # mu = torch.from_numpy(best_gmm.means_)
        # cov = torch.from_numpy(best_gmm.covariances_)
        #
        # w, order = w.sort(0, descending=True)
        # mu = torch.gather(mu, 0, order.view(-1, 1).expand_as(mu))
        # cov = torch.gather(cov, 0, order.view(-1, 1).expand_as(cov))
        #
        # for i in range(w.size(0)):
        #     cluster_list[spkr_id].append([w[i], mu[i], cov[i]])

        # uncomment this to save mean only
        w = torch.from_numpy(best_gmm.weights_)
        mu = torch.from_numpy(best_gmm.means_)

        # w, order = w.sort(0, descending=True)
        # mu = torch.gather(mu, 0, order.view(-1, 1).expand_as(mu)).cpu().numpy()

        for i in range(w.size(0)):
            if w[i] >= 0.01: # threshold to remove outlier
                cluster_list.append(mu[i])
        
        # pick N most distinct style
        gst_idx_list = []
        cluster = np.mean(style_vec_list, axis=0)
        remaining = np.stack(cluster_list, axis=0)
        for _ in range(n_cluster):
            cos_distance = metrics.pairwise.euclidean_distances(cluster, remaining)
            cos_mean = np.array(
                [mean if i not in gst_idx_list else 0 for i, mean in enumerate(np.mean(cos_distance, axis=0))])
            gst_idx = cos_mean.argsort()[::-1][0]
            cluster = np.concatenate((cluster, remaining[gst_idx].reshape(1, -1)))
            gst_idx_list.append(gst_idx)
        
        cluster_list = [cluster[i] for i in range(1, cluster.shape[0])]

    elif mode == 'vblda':
        # parameter
        n_iter = 50
        cluster_list_tmp = []

        # initialize
        X_sty = np.concatenate(style_vec_list, axis=0)
        X = np.concatenate(style_vec_list, axis=0)
        # X = np.concatenate(att_weights_list, axis=0)
        init_ncomp = max(len(style_vec_list) // 50, n_cluster)
        vbgmm = mixture.BayesianGaussianMixture(n_components=init_ncomp, max_iter=2000, random_state=42)

        # Fit using style_token_weight since the dimension of style_vector is too large.
        Y = vbgmm.fit_predict(X)

        # iter vbgmm - lda
        for i in range(n_iter):
            # stage1 : find discriminative feature space
            K = len(set(Y))
            lda = LinearDiscriminantAnalysis()
            X_new = lda.fit_transform(X, Y)

            # stage2 : find cluster labels
            vbgmm = mixture.BayesianGaussianMixture(n_components=K, weight_concentration_prior=0.01, max_iter=1000, random_state=42)
            Y_new = vbgmm.fit_predict(X_new)

            # termination condition
            if len(set(Y_new)) < n_cluster : 
                break
            if np.array_equal(Y, Y_new):
                Y = Y_new
                break

            Y = Y_new

        # find centroid
        clf = NearestCentroid()
        clf.fit(X_sty, Y)
        for i in range(len(set(Y))):
            cluster_list_tmp.append(clf.centroids_[i])

        # pick N most distinct style
        gst_idx_list = []
        cluster = np.mean(style_vec_list, axis=0)
        remaining = np.stack(cluster_list_tmp, axis=0)
        for _ in range(n_cluster):
            cos_distance = metrics.pairwise.cosine_distances(cluster, remaining)
            cos_mean = np.array(
                [mean if i not in gst_idx_list else 0 for i, mean in enumerate(np.mean(cos_distance, axis=0))])
            gst_idx = cos_mean.argsort()[::-1][0]
            cluster = np.concatenate((cluster, remaining[gst_idx].reshape(1, -1)))
            gst_idx_list.append(gst_idx)

        cluster_list = [cluster[i] for i in range(1, cluster.shape[0])]

    print(log_txt)
    return cluster_list


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()