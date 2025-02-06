import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_file", type=str, default="/home/yueyulin/data/MetaMathQA-395K.json")
    parser.add_argument("--output_jsonl_file", type=str, default="/home/yueyulin/data/MetaMathQA-395K-processed.jsonl")
    args = parser.parse_args()
    import json
    with open(args.input_json_file, 'r') as f:
        data = json.load(f)
    print(len(data))
    with open(args.output_jsonl_file, 'w') as f:
        for item in data:
            """
            {"query": "If John's camera broke and he decided to rent a $5000 camera for 4 weeks, with a rental fee of 10% of the camera's value per week, and his friend agreed to pay 40% of the rental fee, how much did John have to pay?", "response": "The rental fee per week is 10% of $5000, which is $5000 * 10% = $500\nSince John rented the camera for 4 weeks, the total rental fee is $500 * 4 = $2000\nJohn's friend agreed to pay 40% of the rental fee, which is $2000 * 40% = $800\nTherefore, John had to pay $2000 - $800 = $1200\n#### 1200\nThe answer is: 1200", "type": "GSM_Rephrased", "original_question": "John's camera broke so he decided to rent one for 4 weeks.  It was a $5000 camera and the rental fee was 10% of the value per week.  His friend who was there when it broke agreed to pay 40% of the rental fee.  How much did John pay?"}
            """
            system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process are enclosed within <think> </think>, i.e., <think> reasoning process here </think>**Final Answer:**\nanswer here. "
            query = item['query']
            response = item['response']
            #find the ground truth
            index_of_answer = response.find("The answer is: ")
            if index_of_answer == -1:
                print(f"no answer found in {response}")
                break
            ground_truth = response[index_of_answer + len("The answer is: "):]
            json_str = json.dumps({"problem": query, "ground_truth": ground_truth, "prompt": [{"role":"system","content":system_prompt},{"role": "user", "content": query}]},ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
        