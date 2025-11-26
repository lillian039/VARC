import json
from pathlib import Path
import argparse
from utils.eval_utils import get_majority_vote
from utils.html_vis_support import render_results_html
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--html-output",
    type=Path,
    default="analysis_results_arc.html",
    help="Path to save the HTML analysis results.",
)

parser.add_argument(
    "--output-root",
    type=str,
    required=True,
)

parser.add_argument(
    "--task-type",
    type=str,
    default="ARC-AGI",
)
args = parser.parse_args()

HTML_OUTPUT = args.html_output

if args.task_type == "ARC-AGI":
    tasks = ['af24b4cc', 'e1d2900e', '903d1b4a', '4e469f39', 'b1fc8b8e', '2c737e39', '992798f6', '00576224', '48131b3c', '60a26a3e', '59341089', '31d5ba1a', 'e633a9e5', '62ab2642', '73c3b0d8', 'c663677b', 'c48954c1', '08573cc6', '136b0064', '929ab4e9', '5b526a93', 'ef26cbf6', 'fafd9572', '67c52801', 'ad7e01d0', '506d28a5', '27a77e38', 'd492a647', '72a961c9', 'fd4b2b02', 'bf89d739', 'f5aa3634', 'b942fd60', 'd282b262', '9772c176', 'ed74f2f2', '184a9768', '94133066', '256b0a75', 'e681b708', 'ce8d95cc', '817e6c09', '7d18a6fb', '1da012fc', '310f3251', 'bf699163', '917bccba', '551d5bf1', 'b457fec5', '50a16a69', '7953d61e', '9a4bb226', 'c97c0139', 'd4b1c2b1', '1d398264', '29700607', '8597cfd7', 'a59b95c0', '2f0c5170', 'bd14c3bf', '9caba7c3', 'e6de6e8f', 'da515329', '31adaf00', 'f5c89df1', 'be03b35f', '833dafe3', '6ea4a07e', '2546ccf6', '21f83797', '696d4842', 'd94c3b52', '4ff4c9da', '3979b1a8', 'bf32578f', 'd304284e', 'c62e2108', 'b0722778', 'd19f7514', '358ba94e', 'd017b73f', '4c177718', 'b7999b51', 'e345f17b', 'e4075551', '50aad11f', '66e6c45b', 'c074846d', '0b17323b', '4b6b68e5', '84db8fc4', 'ff72ca3e', '8ee62060', '52fd389e', 'ae58858e', 'fea12743', '0f63c0b9', 'e99362f0', '195ba7dc', 'f3cdc58f', 'a8610ef7', 'e760a62e', 'aa300dc3', 'ea9794b1', 'e41c6fd3', '5d2a5c43', 'e66aafb8', 'ca8de6ea', '19bb5feb', '7c8af763', 'e872b94a', '6f473927', 'ac605cbb', 'ac3e2b04', '0e671a1a', 'ac0c5833', 'fb791726', '351d6448', 'ce039d91', '45bbe264', '332efdb3', 'c64f1187', '5b6cbef5', '1d0a4b61', '42918530', '7bb29440', '3a301edc', '896d5239', '505fff84', 'cfb2ce5a', '140c817e', '69889d6e', '20818e16', '9b2a60aa', '626c0bcc', 'a57f2f04', '477d2879', '05a7bcf2', '81c0276b', 'ba9d41b8', 'e133d23d', '604001fa', '3ee1011a', '85b81ff1', '17b80ad2', '9b365c51', 'e7dd8335', '2a5f8217', '712bf12e', '84f2aca1', 'ac2e8ecf', 'e2092e0c', '33b52de3', '5833af48', '319f2597', 'aa18de87', 'cb227835', 'e74e1818', '15663ba9', 'b4a43f3b', '281123b4', 'fc754716', 'e5790162', '94414823', '642d658d', '96a8c0cd', '2697da3f', 'e9b4f6fc', 'bcb3040b', '55783887', '1acc24af', '981571dc', '705a3229', '1c02dbbe', 'ca8f78db', '1e97544e', '92e50de0', 'e57337a4', '4852f2fa', '7d1f7ee8', 'e1baa8a4', '14754a24', '62b74c02', '7d419a02', '94be5b80', '68b67ca3', '2072aba6', 'fe9372f3', '137f0df0', 'c6e1b8da', '16b78196', '1c0d0a4b', 'f0afb749', 'de493100', '1990f7a8', '423a55dc', '2753e76c', 'f21745ec', 'bc4146bd', '79fb03f4', '3d31c5b3', 'c35c1b4c', 'cf133acc', 'da2b0fe3', '15696249', '0c9aba6e', 'e7a25a18', 'd5c634a2', '414297c0', '009d5c81', '0becf7df', 'f3e62deb', '58743b76', '9b4c17c4', '891232d6', '4e45f183', 'c7d4e6ad', 'a04b2602', 'd37a1ef5', '25094a63', '0c786b71', 'f3b10344', 'b7f8a4d8', 'b7cb93ac', 'b15fca0b', '1a6449f1', '67636eac', 'f823c43c', '27f8ce4f', 'fd096ab6', '0a1d4ef5', '6a11f6da', '8fbca751', '6ad5bdfd', '3ed85e70', '09c534e7', '642248e4', '9f27f097', '50f325b5', '88207623', '45737921', '11e1fe23', 'aee291af', '90347967', 'e7b06bea', '03560426', 'e7639916', 'e21a174a', '4aab4007', 'c658a4bd', '5783df64', '1c56ad9f', 'c1990cce', '93c31fbe', '5af49b42', '7e02026e', '2685904e', 'c8b7cc0f', '15113be4', 'b20f7c8b', '575b1a71', 'e0fb7511', '3b4c2228', '32e9702f', 'ccd554ac', 'af22c60d', 'bbb1b8b6', 'f9d67f8b', '0d87d2a6', 'ed98d772', '9ddd00f0', '070dd51e', '9356391f', '4acc7107', '47996f11', '8dae5dfc', 'e5c44e8f', 'e9bb6954', 'dc2aa30b', 'd2acf2cb', '292dd178', 'f9a67cb5', '20981f0e', '12997ef3', '103eff5b', '770cc55f', '0692e18c', '8719f442', 'e88171ec', '95a58926', '639f5a19', '40f6cd08', '3f23242b', 'd47aa2ff', '5b692c0f', 'a934301b', 'a3f84088', '72207abc', '73182012', '0a2355a6', 'a680ac02', '58e15b12', '64a7c07e', 'e9c9d9a1', '12eac192', 'dc2e9a9d', 'ecaa0ec1', '4cd1b7b2', '7c9b52a0', '3391f8c0', '9c56f360', '0607ce86', '97239e3d', 'a406ac07', 'baf41dbf', 'c3202e5a', '13713586', '42a15761', 'e619ca6e', 'e69241bd', 'd56f2372', '5207a7b5', '8ba14f53', 'b0f4d537', 'aa4ec2a5', '3490cc26', '9def23fe', 'd4c90558', '12422b43', '99306f82', '516b51b7', 'cad67732', 'f4081712', '212895b5', '67b4a34d', '845d6e51', '66f2d22f', '73ccf9c2', 'f83cb3f6', 'd931c21c', '17cae0c1', '22a4bbc2', '0934a4d8', '8cb8642d', '1e81d6f9', '85fa5666', '0bb8deee', '55059096', '4364c1c4', '4f537728', 'aab50785', '9110e3c5', '762cd429', '7039b2d7', 'bb52a14b', 'ea959feb', 'dd2401ed', '48f8583b', '2b01abd0', 'b7fb29bc', '8b28cd80', '782b5218', '5a5a2103', '759f3fd3', 'e9ac8c9e', '9c1e755f', '37d3e8b2', '9bebae7a', '1a2e2828', '34b99a2b', '60c09cac', 'f45f5ca7', '7ee1c6ea', '54db823b', '00dbd492', 'b9630600', '2037f2c7', 'c87289bb', '692cd3b6', '79369cc6', 'c92b942c', '6df30ad6', 'a096bf4d', 'e78887d1', '5ffb2104', '5289ad53', '8a371977', 'f0df5ff0', 'df8cc377', 'cd3c21df', '963f59bc', '3194b014', '93b4f4b3', '2c0b0aff', '456873bc', '8e2edd66', 'f8be4b64', 'e95e3d8e', '695367ec', '18419cfa']
else:
    tasks = ['65b59efc', '2d0172a1', '7b0280bc', 'e12f9a14', 'e8686506', '88e364bc', '7b3084d4', 'a251c730', 'd35bdbdc', 'fc7cae8d', '8f3a5a89', '35ab12c3', '88bcf3b4', 'cb2d8a2c', 'b10624e5', '135a2760', '5dbc8537', 'e87109e9', '4a21e3da', 'abc82100', '64efde09', '3e6067c3', '2ba387bc', '5961cc34', '38007db0', '20270e3b', '142ca369', '67e490f4', 'bf45cf4b', 'dbff022c', '4e34c42c', '36a08778', '7c66cb00', 'a395ee82', '271d71e2', 'f931b4a8', 'faa9f03d', '581f7754', 'a25697e4', '71e489b6', '21897d95', '8698868d', '1818057f', '6e4f6532', '9385bd28', '0934a4d8', '4c3d4a41', '20a9e565', 'db695cfb', '78332cb0', '80a900e0', 'a47bf94d', '800d221b', 'aa4ec2a5', 'a6f40cea', '13e47133', 'f560132c', '8b9c3697', '8f215267', '247ef758', 'de809cff', 'b99e7126', 'd8e07eb2', '62593bfd', 'e376de54', '16de56c4', '4c7dc4dd', '7491f3cf', 'b9e38dc0', '221dfab4', '332f06d7', '45a5af55', 'eee78d87', '3dc255db', 'a32d8b75', '4c416de3', 'edb79dae', '136b0064', 'b5ca7ac4', '6e453dd6', '7b80bb43', 'b6f77b65', '6ffbe589', 'dd6b8c4b', '28a6681f', '7666fa5d', '2c181942', '7b5033c1', 'c4d067a0', 'da515329', '446ef5d2', '269e22fb', '409aa875', 'e3721c99', '291dc1e1', 'db0c5428', '8b7bacbf', '58490d8a', '5545f144', '2b83f449', '9aaea919', 'dfadab01', '89565ca0', '53fb4810', '31f7f899', '58f5dbd5', '1ae2feb7', 'b0039139', 'c7f57c3e', '3a25b0d8', '16b78196', '9bbf930d', '898e7135', '7ed72f31', 'cbebaa4b', '97d7923e', 'd59b0160', '8e5c0c38', '981571dc', '195c6913']

# Load predictions
answer_set = {}
for name in tasks:
    roots = args.output_root.split(',')
    data = []
    for root in roots:
        root = root.strip()
        cur_task_save = f"{root}/{name}_predictions.json"
        if not os.path.exists(cur_task_save):
           print(f"Warning: Prediction file for task {name} not found at {cur_task_save}. Skipping.")
           continue
        with open(cur_task_save, "r") as f:
            data.append(json.load(f))
    
    if len(data) == 0:
        continue
    all_data = {k: [] for k in data[0].keys()}
    for d in data:
        for k in data[0].keys():
            all_data[k] = all_data[k] + d[k]

    answer_set[name] = all_data

def analyze_data(answer_set, task_names, task_type, html_output: Path = HTML_OUTPUT):
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
    random_correct_score = 0

    for task_name in task_names:
        if task_name not in answer_set:
            print(f"Warning: No answers found for task {task_name}. Skipping.")
            continue
        all_task_num += 1
        for cur_index in answer_set[task_name]:
            if cur_index == 'answer':
                continue
            majority_vote = get_majority_vote(answer_set[task_name][cur_index])
            random_sampled_results = answer_set[task_name][cur_index][0]
            ground_truth = ground_truths.get(task_name, {}).get(cur_index)
            
            if random_sampled_results == ground_truth:
                random_correct_score += 1 / len(answer_set[task_name])
            has_ground_truth = ground_truth is not None
            task_payload = tasks_payload[task_name]
            example_payload = task_payload["examples"].setdefault(
                cur_index, {"input": None, "answer": ground_truth, "majority_vote": []}
            )
            majority_entries = []
            for entry in majority_vote[:2]:
                matches_answer = has_ground_truth and entry["prediction"] == ground_truth
                majority_entries.append(
                    {
                        "prediction": entry["prediction"],
                        "votes": entry["votes"],
                        "matches_answer": matches_answer,
                    }
                )
            example_payload["majority_vote"] = majority_entries
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

    print(all_task_num)

    pass_at_1_score = correct_num_1 / all_task_num
    pass_at_2_score = correct_num_2 / all_task_num
    oracle_score = correct_oracle_num / all_task_num
    metrics = {
        "pass_at_1": pass_at_1_score,
        "pass_at_2": pass_at_2_score,
        "oracle": oracle_score,
        "total_tasks": all_task_num,
    }

    oracle_rank_sorted = dict(sorted(oracle_rank.items()))
    sum_correct = 0
    for rank, count in oracle_rank_sorted.items():
        print(f"Oracle rank {rank}: {count}. Cumulative: {(sum_correct + count) / (all_task_num):.4f}")
        sum_correct += count
    
    print(f"Final Oracle Score: {oracle_score:.4f}")
    print(f"Final Pass@1 Score: {pass_at_1_score:.4f}")
    print(f"Final Pass@2 Score: {pass_at_2_score:.4f}")
    render_results_html(tasks_payload, metrics, html_output)
    print(f"HTML visualization saved to {html_output.resolve()}")

analyze_data(answer_set, tasks, args.task_type)
