import json
from utils.eval_utils import get_majority_vote

def analyze_data(answer_set, task_names, task_type):
    oracle_rank = {}
    ground_truths = {}
    tasks_payload = {}
    for task_name in task_names:
        with open(f'raw_data/{task_type}/data/evaluation/{task_name}.json', 'r') as f:
            cur_data = json.load(f)
        test_data = cur_data['test']
        ground_truths[task_name] = {}
        train_examples = [
            {"input": item.get("input"), "output": item.get("output")}
            for item in cur_data.get("train", [])
        ]
        tasks_payload[task_name] = {"examples": {}, "train_examples": train_examples}
        for idx, item in enumerate(test_data):
            example_id = str(idx)
            ground_truths[task_name][example_id] = item['output']
            tasks_payload[task_name]["examples"][example_id] = {
                "input": item.get("input"),
                "answer": item.get("output"),
                "majority_vote": [],
            }

    all_task_num, correct_num_1, correct_num_2, correct_oracle_num = 0, 0, 0, 0

    for task_name in task_names:
        all_task_num += 1
        for cur_index in answer_set[task_name]:
            majority_vote = get_majority_vote(answer_set[task_name][cur_index])
            cur_index = str(cur_index)
            ground_truth = ground_truths[task_name][cur_index]

            pass_at_1 = (len(majority_vote) > 0 and majority_vote[0]["prediction"] == ground_truth)
            if len(majority_vote) > 1:
                pass_at_2 = (majority_vote[1]["prediction"] == ground_truth or pass_at_1)
            else:
                pass_at_2 = pass_at_1
            cur_score = 1 / len(answer_set[task_name])

            if pass_at_1:
                correct_num_1 += cur_score
                correct_num_2 += cur_score
            elif pass_at_2:
                correct_num_2 += cur_score

            oracle_result = False
            for rank, entry in enumerate(majority_vote):
                if entry['prediction'] == ground_truth:
                    oracle_result = True
                    if rank + 1 not in oracle_rank:
                        oracle_rank[rank + 1] = 0
                    oracle_rank[rank + 1] += cur_score
                    break

            if oracle_result:
                correct_oracle_num += cur_score


    pass_at_1_score = correct_num_1 / all_task_num
    pass_at_2_score = correct_num_2 / all_task_num
    oracle_score = correct_oracle_num / all_task_num
   
    oracle_rank_sorted = dict(sorted(oracle_rank.items()))
    sum_correct = 0
    for rank, count in oracle_rank_sorted.items():
        print(f"Oracle rank {rank}: {count}. Cumulative: {(sum_correct + count) / (all_task_num):.4f}")
        sum_correct += count
    
    print(f"Final Oracle Score: {oracle_score:.4f}")
    print(f"Final Pass@1 Score: {pass_at_1_score:.4f}")
    print(f"Final Pass@2 Score: {pass_at_2_score:.4f}")
    if pass_at_1_score or pass_at_2_score:
        print(f"Current task {task_names[0]} is Correct!✅")
    else:
        print(f"Current task {task_names[0]} is Wrong!❌")
