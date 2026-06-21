import pandas as pd
from collections import deque   
MAX_CONVO_LEN=10
MAX_HISTORY_LEN=6



def generate_attack(evaluator,attacker,target,attacker_system_prompt):
    system_message = attacker_system_prompt
    max_convo_len=MAX_CONVO_LEN
    history = deque(maxlen=MAX_HISTORY_LEN)
    feed_back_message=None
    convo_list=[]
    data=None
    messages=None
    is_successful_attack=False

    while max_convo_len:
        try:
            if history:
                messages = [system_message]+list(history)
            else:
                messages = [system_message]

            user_message_content="generate an attack following the instructions in the system prompt"
            if feed_back_message:
                user_message_content += "this is the judge's feedback from previous iteration: " + feed_back_message
            # attacker 
            attack=attacker.invoke(messages+[{"role": "user", "content": user_message_content}])
            attack=attack.content.strip()
            # target
            target_response = target.chat_secure(attack,history=history)
            history.append({"role": "user", "content": attack})
            history.append({"role": "assistant", "content": target_response["reply"]})
            # evaluation
            evaluation_result=0
            judge_output={"reason": "Blocked"}
            if not target_response["blocked"]:
                evaluation_result, judge_output = evaluator.evaluate_response_with_feedback(attack,target_response["reply"])
            data={"attack":attack,
                "target_response": target_response["reply"],
                "judge_reason": judge_output["reason"],                
                "remaining":max_convo_len,
                "result":evaluation_result}
            convo_list.append(data)
            # exit if successful attack
            if judge_output["result"]==1:
                is_successful_attack=True
                break
            max_convo_len-=1
        except:
            break

    # append the conversation
    if len(convo_list)>0:
        try:
            df=pd.read_csv("./data_generation/custom/outputs/attack_generation.csv")
            current_dataset = pd.concat([df, pd.DataFrame(convo_list)], ignore_index=True)
        except FileNotFoundError:
            current_dataset=pd.DataFrame(convo_list)
        current_dataset.to_csv("./data_generation/custom/outputs/attack_generation.csv", index=False)
    return data,is_successful_attack


