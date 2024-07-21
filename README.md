# OpenAGI: Package for AI Agent Creation
<a href='https://arxiv.org/abs/2304.04370'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](https://github.com/agiresearch/OpenAGI/blob/main/LICENSE)
<a href='https://discord.gg/B2HFxEgTJX'><img src='https://img.shields.io/badge/Community-Discord-8A2BE2'></a>


## ✈️ Getting Started
OpenAGI is used as the agent creation package to build agents for [AIOS](https://github.com/agiresearch/AIOS).
### Installation
From PyPI
```
pip install pyopenagi
```
Locally
```
git clone https://agiresearch/OpenAGI
cd OpenAGI
pip install -e .
```

### Usage

#### Add a new agent
To add a new agent, first you need to create a folder under the pyopenagi/agents folder.
The folder needs to be the following structure:
```
- pyopenagi/agents
  - author
    - agent_name
      - agent.py # main code for the agent execution logic
      - config.json # set up configurations for agent
      - meta_requirements.txt # dependencies that the agent needs
```
If you want to use external tools provided by openagi in your agents, you can follow instructions of setting up tools in [How to setup external tools](./tools.md).
If you want to add new tools for your developing agent,
you need to add a new tool file in the [folder](./pyopenagi/tools/).

#### Upload agent
If you have developed and tested your agent, and you would like to share your agents, you can use the following to upload your agents
```
python pyopenagi/agents/interact.py --mode upload --agent <author_name/agent_name>
```
💡Note that the `agent` param must exactly match the folder you put your agent locally.

#### Download agent
If you want to look at implementations of other agents that others have developed, you can use the following command:
```
python pyopenagi/agents/interact.py --mode download --agent <author_name/agent_name>
```


## 🚀 Contributions
For detailed information on how to contribute, see [CONTRIBUTE](./CONTRIBUTE.md). If you would like to contribute to the codebase, [issues](https://github.com/agiresearch/OpenAGI/issues) or [pull requests](https://github.com/agiresearch/OpenAGI/pulls) are always welcome!

## 🖋️ Research
Please check out our [implementation](./research) for our research paper [OpenAGI: When LLM Meets Domain Experts](https://arxiv.org/abs/2304.04370).

```
@article{openagi,
  title={OpenAGI: When LLM Meets Domain Experts},
  author={Ge, Yingqiang and Hua, Wenyue and Mei, Kai and Ji, Jianchao and Tan, Juntao and Xu, Shuyuan and Li, Zelong and Zhang, Yongfeng},
  journal={In Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## 🌍 OpenAGI Contributors
[![OpenAGI contributors](https://contrib.rocks/image?repo=agiresearch/OpenAGI&max=300)](https://github.com/agiresearch/OpenAGI/graphs/contributors)



## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=agiresearch/OpenAGI&type=Date)](https://star-history.com/#agiresearch/OpenAGI&Date)
