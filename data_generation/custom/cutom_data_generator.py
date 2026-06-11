import pandas as pd

from utils import generate_attack
# from evaluation.harm_detection import judge

class CustomGenerator:
    def __init__(self, attack_type,harm_type,judge, attacker, target):
        # llms names types
        self.attack_type = attack_type
        self.harm_type = harm_type
        self.judge = judge
        self.attacker = attacker
        self.target = target

        # TODO: add llms here 


    def generate_dataset(self, num_samples):
        # TODO: add system prompts
        target_system_message=None
        attacker_system_prompt=None

        dataset = []
        for i in range(num_samples):
            result = generate_attack(target=self.target,judge=self.judge,\
                                    attacker=self.attacker,target_system_message=target_system_message,\
                                    attacker_system_prompt=attacker_system_prompt)

            if result:
                dataset.append(result)

            if (i + 1) % 2 == 0:
                current_dataset = pd.DataFrame(dataset)

                try:
                    df = pd.read_csv("jailbreak_dataset.csv")
                    current_dataset = pd.concat([df, current_dataset], ignore_index=True)
                except FileNotFoundError:
                    pass

                current_dataset.to_csv("jailbreak_dataset.csv", index=False)
                dataset = []

        # save remaining samples
        if dataset:
            current_dataset = pd.DataFrame(dataset)
            try:
                df = pd.read_csv("jailbreak_dataset.csv")
                current_dataset = pd.concat([df, current_dataset], ignore_index=True)
            except FileNotFoundError:
                pass

            current_dataset.to_csv("jailbreak_dataset.csv", index=False)

        return dataset