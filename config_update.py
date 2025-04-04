import wandb

api = wandb.Api()

run = api.run("hyeonseong03-hanyang-university/IAM/awnj6obt")
run.config["scheduler"] = "stepLR"
run.update()  # 저장