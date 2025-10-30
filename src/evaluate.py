import json 
import numpy as np

CLASSES = ['far_table_bounce', 'close_table_bounce', 'far_table_serve', 'close_table_serve', 
           'far_table_forehand', 'close_table_forehand', 'far_table_backhand', 'close_table_backhand']


def convert_gold_data(gold_file):
    with open(gold_file, 'r') as gf:
        gold_data = json.load(gf)
    
    converted = {}
    video_labels = gold_data[0]['videoLabels']
    for label in video_labels:
        frame_index = label['ranges'][0]['start']
        event = label['timelinelabels'][0]

        converted[frame_index] = {
            'frame_index': frame_index-1,
            'event_type': event,
        }
    return converted


def calcualte_ap(event_type, predictions, gold_data, frame_tolerance=5):
    predictions.sort(key=lambda x: x["score"], reverse=True)


    gt_used = np.zeros(len(gold_data), dtype=bool)
    tp, fp = [], []

    wrongly_detected = []
    not_detected = []


    for pred in predictions:
        matched = False
        for i, gt in enumerate(gold_data):
            if not gt_used[i] and abs(pred['frame_index'] - gt['frame_index']) <= frame_tolerance:
                matched = True
                gt_used[i] = True
                break
        tp.append(1 if matched else 0)
        fp.append(0 if matched else 1)
        if not matched:
            # adding false positive for predictions means that do not match any ground truth
            wrongly_detected.append(pred['frame_index'])

    # add remaining false positives for ground truths that were not detected
    # this is not necessary for AP calculation but for completeness
    for i in range(len(gold_data)):
        if not gt_used[i]:
            not_detected.append(gold_data[i]['frame_index']) 

    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall = tp_cum / len(gold_data)
    AP = np.trapezoid(precision, recall)

    print(f"Precision: {precision[-1]:.4f}, Recall: {recall[-1]:.4f}")
    
    wrongly_detected.sort()
    not_detected.sort()
    print(f"Wrongly detected frames for {event_type}: {wrongly_detected}")
    print(f"Not detected frames for {event_type}: {not_detected}")


    return AP

def evaluate_performance(pred_file, gold_file):
    with open(pred_file, 'r') as pf, open(gold_file, 'r') as gf:
        predictions = json.load(pf)
        gold_data = json.load(gf)
    
    converted_gold = convert_gold_data(gold_file)
    
    total_AP = []
    
    for event_type in CLASSES:
        predictions_this_class = {}
        gold_data_this_class = {}
        for i, (fid, event) in enumerate(predictions.items()):
            if event['event_type'] == event_type:
                predictions_this_class[i] = {
                    'frame_index': int(event['frame_index']),
                    'event_type': event['event_type'],
                    'score': event['score']
                }
        for j, (fid, event) in enumerate(converted_gold.items()):
            if event['event_type'] == event_type:
                gold_data_this_class[j] = {
                    'frame_index': int(event['frame_index']),
                    'event_type': event['event_type']
                }

        predictions_this_class = list(predictions_this_class.values())
        gold_data_this_class = list(gold_data_this_class.values())
        
        ap_this_class = calcualte_ap(event_type, predictions_this_class, gold_data_this_class, frame_tolerance=3)
        
        total_AP.append(ap_this_class)
        print(f"AP for {event_type}: {ap_this_class:.4f} \n")
    
    mAP = np.mean(total_AP)
    print(f"Mean AP: {mAP:.4f}")
        
        
       

    
    




if __name__ == "__main__":
    pred_file = '/home/august/github/TTA_Project/src/predicted_events_25WPF_TPE_M11_G_Chen_Po_Yen_TPE_v_Murakami_JPN_game1.json'
    gold_file = '/home/august/github/TTA_Project/data/labels/25WPF_TPE_M11_G_Chen_Po_Yen_TPE_v_Murakami_JPN_game1.json'
    evaluate_performance(pred_file, gold_file)