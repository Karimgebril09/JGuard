import pandas as pd
from collections import deque   
from langchain.messages import SystemMessage, HumanMessage, AIMessage



def generate_attack(judge,attacker,target,target_system_message,attacker_system_prompt):
    print("start generating an attack...")
    system_message = attacker_system_prompt
    max_convo_len=10
    history = deque(maxlen=5)
    feed_back_message=None
    convo_list=[]
    data=None
    while max_convo_len:
        try:
            messages = [system_message]+list(history) if history else [system_message]
            user_message_content="generate an attack following the instructions in the system prompt"
            if feed_back_message:
                user_message_content += "this is the judge's feedback from previous iteration: " + feed_back_message
            attack=attacker.invoke(messages+[HumanMessage(content=user_message_content)])
            attack=attack.content.strip()

            target_response = target.invoke([target_system_message,HumanMessage(content=attack)])

            history.append(HumanMessage(content=attack))
            history.append(AIMessage(content=target_response.content))

            judge_output = judge.run(user_prompt=attack, llm_response=target_response.content, max_trials=3, default_val=0)
            feed_back_message = judge_output["reason"]

            data={"attack":attack,
                "target_response": target_response.content,
                "judge_reason": judge_output["reason"],                
                 "remaining":max_convo_len,
                  "result":judge_output["result"]
               }
            convo_list.append(data)
            if judge_output["result"]==1:
                break
            max_convo_len-=1
        except:
            break

    if len(convo_list)>0:
        try:
            df=pd.read_csv("attack_generation.csv")
            current_dataset = pd.concat([df, pd.DataFrame(convo_list)], ignore_index=True)
        except FileNotFoundError:
            current_dataset=pd.DataFrame(convo_list)
        current_dataset.to_csv("attack_generation.csv", index=False)
    print("finished generating attack")
    return data


