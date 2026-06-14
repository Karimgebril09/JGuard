import pandas as pd
from collections import deque   


def generate_attack(evaluator,attacker,target,attacker_system_prompt):
    print("start generating an attack...")
    system_message = attacker_system_prompt
    max_convo_len=10
    history = deque(maxlen=6)
    feed_back_message=None
    convo_list=[]
    data=None
    while max_convo_len:
        try:
            messages = [system_message]+list(history) if history else [system_message]
            user_message_content="generate an attack following the instructions in the system prompt"
            if feed_back_message:
                user_message_content += "this is the judge's feedback from previous iteration: " + feed_back_message
            attack=attacker.invoke(messages+[{"role": "user", "content": user_message_content}])
            attack=attack.content.strip()
            target_response = target.chat_secure(attack,history=history)
            history.append({"role": "user", "content": attack})
            history.append({"role": "assistant", "content": target_response["reply"]})
            evaluation_result=0
            judge_output={"reason": "Blocked"}
            if not target_response["blocked"]:
                evaluation_result, judge_output = evaluator.evaluate_response_with_feedback(attack,target_response["reply"])
            print(f"evaluation result: {evaluation_result}, judge reason: {judge_output['reason']}")
            data={
                "attack":attack,
                "target_response": target_response["reply"],
                "judge_reason": judge_output["reason"],                
                "remaining":max_convo_len,
                "result":evaluation_result
                }
            convo_list.append(data)
            if judge_output["result"]==1:
                break
            max_convo_len-=1
        except:
            break

    if len(convo_list)>0:
        try:
            df=pd.read_csv("./data_generation/custom/outputs/attack_generation.csv")
            current_dataset = pd.concat([df, pd.DataFrame(convo_list)], ignore_index=True)
        except FileNotFoundError:
            current_dataset=pd.DataFrame(convo_list)
        current_dataset.to_csv("./data_generation/custom/outputs/attack_generation.csv", index=False)
    print("finished generating attack")
    return data


