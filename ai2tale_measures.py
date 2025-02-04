from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import re
import os
import json
import scipy.io as io
import pickle
from tqdm import tqdm
from nltk import word_tokenize
from sklearn.metrics import f1_score
from sklearn import metrics as mt

import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def check_in_dictionary(sentence, dictionary):
    count_exist = 0
    for word in sentence.split():
        if word not in dictionary:
            if stemmer.stem(word) in dictionary:
                count_exist += 1
        else:
            count_exist += 1
    return count_exist


def load_vocab(dir_path):
    ind_word_vocab_size_path = dir_path + 'ind_word_vocab_size_m' + '.mat'
    ind_word_vocab_size = io.loadmat(ind_word_vocab_size_path)
    vocab_size = ind_word_vocab_size['vocab_size'][0][0]
    return vocab_size


def load_data(dir_path):
    with open(dir_path + 'ind_word.pickle', 'rb') as f:
        word_index = pickle.load(f)
    return word_index


def to_text(data, index_word):
    text_list = []
    for idata in data:
        text = ''
        for jdata in idata:
            for kdata in jdata:
                text += index_word[int(kdata)] + ' '
            text += ":::"
        text_list.append(text.strip())
    return text_list


def to_text_selected_data(selected_ids, x_data, word_index):
    text_list = []
    for idx, ids in enumerate(selected_ids):
        text = ''
        for i_ids in ids:
            for kdata in x_data[idx][i_ids]:
                text += word_index[int(kdata)] + ' '
            text += "\n"
        text_list.append(text)
    return text_list


def clean_str_yk(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " s", string)
    string = re.sub(r"\'ve", " ve", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " re", string)
    string = re.sub(r"\'d", " d", string)
    string = re.sub(r"\'ll", " ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\\", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def word_tokenize_data(idata):
    return word_tokenize(clean_str_yk(idata.strip()).lower())


def stop_stem_words(data):
    data_list_flattern = [stemmer.stem(w) for w in data if not w in stop_words]
    return data_list_flattern


def check_cog_diversity(selected_sentence, scarcity_list, consistency_list, reciprocity_list, authority_list, socialproof_list, liking_list, security_list):
    sentence = word_tokenize_data(selected_sentence)
    sentence_tokens = [stemmer.stem(w) for w in sentence if not w in stop_words]

    c_scar = 0
    c_con = 0
    c_rec = 0
    c_aut = 0
    c_soc = 0
    c_lik = 0
    c_sec = 0
    for i_s_tokens in sentence_tokens:
        if i_s_tokens in scarcity_list:
            c_scar += 1
        elif i_s_tokens in consistency_list:
            c_con += 1
        elif i_s_tokens in reciprocity_list:
            c_rec += 1
        elif i_s_tokens in authority_list:
            c_aut += 1
        elif i_s_tokens in socialproof_list:
            c_soc += 1
        elif i_s_tokens in liking_list:
            c_lik += 1
        elif i_s_tokens in security_list:
            c_sec += 1
        else:
            max = 0
    
    check_important = None
    max = np.max(np.array([c_scar, c_con, c_rec, c_aut, c_soc, c_lik, c_sec]))
    if (max == c_scar or max == c_con or max == c_aut):
        check_important = True
    else:
        check_important = False
    
    return max, c_scar, c_con, c_rec, c_aut, c_soc, c_lik, c_sec, check_important


def validate_gt_selected_data(model_path, file_results, args):
    print('Loading dataset...')
    word_index = load_data(args.data)

    if os.path.exists(model_path):
        x_val_selected_ids_process = np.load(
            model_path + 'selected_ids_process_' + str(args.k) + '.npy')

        pred_val = np.load(model_path + 'pred_val_' + str(args.k) + '.npy')
        x_val = np.load(model_path + 'x_val_' + str(args.k) + '.npy')
        y_val = np.load(model_path + 'y_val_' + str(args.k) + '.npy')

        if y_val.shape[-1] != 2:
            if y_val.shape[0] != pred_val.shape[0]:
                y_val = np.reshape(y_val, [-1, 2])
            else:
                y_val = np.eye(2)[y_val.astype(np.int32)]
        if pred_val.shape[-1] != 2:
            pred_val = np.reshape(pred_val, [-1, 2])

        val_acc = np.mean(np.argmax(y_val, axis=-1) == np.argmax(pred_val, axis=-1))
        print('the validation accuracy is {}.'.format(val_acc))
        file_results.write('the validation accuracy is {}.\n'.format(val_acc))

        f1_score_binary = f1_score(np.argmax(y_val, axis=-1), np.argmax(pred_val, axis=-1))
        f1_score_weighted = f1_score(np.argmax(y_val, axis=-1), np.argmax(pred_val, axis=-1), average='weighted')

        print('the validation f1_binary is {}.'.format(f1_score_binary))
        file_results.write('the validation f1_binary is {}.\n'.format(f1_score_binary))

        print('the validation f1_weighted is {}.'.format(f1_score_weighted))
        file_results.write('the validation f1_weighted is {}.\n'.format(f1_score_weighted))

        tn, fp, fn, tp = mt.confusion_matrix(y_true=np.argmax(y_val, axis=-1), y_pred=np.argmax(pred_val, axis=-1)).ravel()
        if (fp + tn) == 0:
            fpr = -1.0
        else:
            fpr = float(fp) / (fp + tn)

        if (tp + fn) == 0:
            fnr = -1.0
        else:
            fnr = float(fn) / (tp + fn)

        print('the validation false positive is {}.'.format(fpr))
        file_results.write('the validation false positive is {}.\n'.format(fpr))

        print('the validation false negative is {}.'.format(fnr))
        file_results.write('the validation false negative is {}.\n'.format(fnr))

        x_val_selected_text_process = to_text_selected_data(x_val_selected_ids_process, x_val, word_index)
        x_val_text = to_text(x_val, word_index)

        with open("phishingdata/kwords_stem.json", 'r') as json_file:
            loaded_dict = json.load(json_file)

        scarcity_list = loaded_dict['scarcity']
        consistency_list = loaded_dict['consistency']
        reciprocity_list = loaded_dict['reciprocity']
        authority_list = loaded_dict['authority']
        socialproof_list = loaded_dict['socialproof']
        liking_list = loaded_dict['liking']
        security_list = loaded_dict['security']
        english_words = loaded_dict['english_words']

        pro_co_cblank = 0
        pro_co_cdump = 0
        pro_wr_cblank = 0
        pro_wr_cdump = 0
        pro_co_cnot_cog = 0
        pro_wr_cnot_cog = 0

        important_sentences = 0
        total_sentences = 0

        y_val = np.argmax(y_val, axis=1).tolist()
        pred_val = np.argmax(pred_val, axis=1).tolist()

        for idx, i_y in enumerate(tqdm(y_val)):
            if i_y == 1:
                if pred_val[idx] == 1:
                    for i_selected_text in x_val_selected_text_process[idx].split("\n"):
                        if i_selected_text.strip() == "":
                            pro_co_cblank += 1
                        else:
                            total_sentences += 1
                            if check_in_dictionary(i_selected_text.strip(), english_words) == 0:
                                pro_co_cdump += 1
                            else:
                                sum, _, _, _, _, _, _, _, _ = check_cog_diversity(
                                    i_selected_text, scarcity_list, consistency_list, reciprocity_list, authority_list, socialproof_list, liking_list, security_list)
                                if sum == 0:
                                    pro_co_cnot_cog += 1
                                else:
                                    important_sentences += 1
                else:
                    for i_selected_text in x_val_selected_text_process[idx].split("\n"):
                        if i_selected_text.strip() == "":
                            pro_wr_cblank += 1
                        else:
                            total_sentences += 1
                            if check_in_dictionary(i_selected_text.strip(), english_words) == 0:
                                pro_wr_cdump += 1
                            else:
                                sum, _, _, _, _, _, _, _, _ = check_cog_diversity(
                                    i_selected_text, scarcity_list, consistency_list, reciprocity_list, authority_list, socialproof_list, liking_list, security_list)
                                if sum == 0:
                                    pro_wr_cnot_cog += 1
                                else:
                                    important_sentences += 1

        if total_sentences == 0:
            tp = 0
        else:
             tp = (1.0*important_sentences)/total_sentences
        
        print("the validation cognitive true positive is " + str(tp))
        file_results.write('the validation cognitive true positive is {}.\n'.format(tp))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./phishingdata/')
    parser.add_argument('--home', type=str, default='./ai2tale_k1_tau0.5/')
    parser.add_argument('--k', type=int, default=1, help='The number of selected sentence in each email data sample')
    parser.add_argument('--lr', type=str, default=0.001, help='The learning rate used in the training process')
    parser.add_argument('--sig', type=float, default=0.3, help='The variance value of the prior distribution mentioned in Eq.(8)')
    parser.add_argument('--lam', type=float, default=0.1, help='The trade-off hyperparameter described in Eq.(9)')
    args = parser.parse_args()

    model_dir = 'sent_models_xai_ai2tale/'
    file_results = open(args.home + model_dir +"file_results_process_all_s" + str(args.k) + ".txt", "w")

    model_path = args.home + model_dir + str(args.sig) + '_' + str(args.lr) + '_' + str(args.lam) + '/'
    file_results.write(model_path + "\n")
    validate_gt_selected_data(model_path, file_results, args)
    
    file_results.close()
