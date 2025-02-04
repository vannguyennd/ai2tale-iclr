import numpy as np


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


def process_sent_ids(sent_ids, x_val, word_index, selected_sents):
    sent_ids_up = []
    x_val_selected_text = to_text_selected_data(sent_ids, x_val, word_index)

    for idx, ids in enumerate(sent_ids):
        sent_ids_i = []
        count = 0
        idx_x_val_selected_text = x_val_selected_text[idx].split("\n")[:-1]
        
        for i_ids in ids:
            if idx_x_val_selected_text[i_ids].strip() != "":
                sent_ids_i.append(i_ids)
                count += 1
                if count == selected_sents:
                    sent_ids_up.append(sent_ids_i)
                    break
        if len(sent_ids_i) < selected_sents:
            count_pad = selected_sents - len(sent_ids_i)
            for _ in range(count_pad):
                sent_ids_i.append(sent_ids_i[0])
            sent_ids_up.append(sent_ids_i)

    return sent_ids_up


def create_dataset_from_score_st_ai2tale_gt(path, scores, x, y, max_epoch, word_index, selected_sents, interp_val, x_val_len):
    """
    Construct data set containing selected sentences by AI2TALE.
    """
    if len(scores.shape) == 3:
        scores = np.squeeze(scores)

    np.save(path + 'scores_' + str(selected_sents) + '.npy', np.array(scores))
    np.save(path + 'x_val_len_' + str(selected_sents) + '.npy', np.array(x_val_len))
    np.save(path + 'x_val_' + str(selected_sents) + '.npy', np.array(x))
    np.save(path + 'y_val_' + str(selected_sents) + '.npy', np.array(y))
    np.save(path + 'pred_val_' + str(selected_sents) + '.npy', np.array(interp_val))

    scores_ids = np.argsort(-scores)
    sent_ids = process_sent_ids(scores_ids, x, word_index, selected_sents)

    np.save(path + 'selected_ids_process_' + str(selected_sents) + '.npy', np.array(sent_ids))

    x_new_process = np.zeros(x.shape)
    for i, sent_id in enumerate(sent_ids):
        x_new_process[i, sent_id, :] = x[i][sent_id]
    np.save(path + 'x_selected_process_' + str(selected_sents) + '.npy', np.array(x_new_process))


def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
