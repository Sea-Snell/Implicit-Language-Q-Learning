from typing import Any, Dict, Iterator, List, Optional, Tuple
from data.language_environment import Language_Environment, Language_Observation, Policy
import requests
import json
from data.rl_data import List_RL_Dataset, Iterable_RL_Dataset, RL_Dataset
import random
from visdial.visdial_base import AnswerEvent, CutoffRule, Event, QuestionEvent, Scene, N_TURNS, StopEvent
from visdial.visdial_base import yn_reward_fs

class VDObservation(Language_Observation):
    def __init__(self, scene: Scene, event: Optional[Event]=None):
        self.scene = scene
        self.event = event

    def add(self, ev: Optional[Event]):
        if self.event is not None:
            ev = self.event.append(ev)
        elif ev is not None:
            ev.scene = self.scene
        return VDObservation(self.scene, ev)
    
    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.event is None:
            return [(self.scene.caption, None)], False
        evs = self.event.get_events()
        sequence = [(self.scene.caption, None)]
        sequence += [(str(evs[i]), evs[i+1].reward if isinstance(evs[i+1], AnswerEvent) else None) for i in range(len(evs)-1)]
        sequence += [(str(evs[-1]), 0.0 if isinstance(evs[-1], StopEvent) else None)]
        terminal = self.event.is_final()
        return sequence, terminal
    
    def __str__(self) -> str:
        if self.event is None:
            return self.scene.caption
        return self.scene.caption+'\n'+'\n'.join(list(map(str, self.event.get_events())))
    
    def metadata(self) -> Optional[Dict[str, Any]]:
        return {'scene': self.scene, 'event': self.event}

class VDEnvironment(Language_Environment):
    def __init__(self, dataset: RL_Dataset, url: str, reward_shift: float=0.0, 
                 reward_scale: float=1.0, actor_stop: bool=False, yn_reward: float=-2.0, 
                 yn_reward_kind: str='none'):
        self.dataset = dataset
        self.remote_env = VDEnvRemoteWrapper(url)
        self.state = self.reset()
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.actor_stop = actor_stop
        self.yn_reward = yn_reward
        self.yn_reward_f = yn_reward_fs[yn_reward_kind]

    def step(self, action: str) -> Tuple[VDObservation, float, bool]:
        if self.state.event is not None and self.state.event.is_final():
            raise Exception("Cannot step after final action")
        if '<stop>' in action and self.actor_stop:
            self.state = self.state.add(StopEvent(0.0, None, None, None))
            reward = 0.0 * self.reward_scale + self.reward_shift
        else:
            self.state = self.state.add(QuestionEvent(action, 0.0, None, None, None))
            response, reward = self.remote_env.step(self.state)
            progress = reward
            if self.state.scene.cutoff_rule is None:
                reward = reward * self.reward_scale + self.reward_shift
            else:
                reward = (-1.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(response) else 0.0)) * self.reward_scale + self.reward_shift
            self.state = self.state.add(AnswerEvent(response, reward, progress, None, None, None))
            if self.state.scene.cutoff_rule is not None and self.state.event.is_final():
                self.state.event.reward = (0.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(response) else 0.0)) * self.reward_scale + self.reward_shift
                reward = (0.0 + (self.yn_reward if self.yn_reward_f is not None and self.yn_reward_f(response) else 0.0)) * self.reward_scale + self.reward_shift
        return self.state, reward, self.state.event.is_final()

    def reset(self) -> VDObservation:
        if isinstance(self.dataset, List_RL_Dataset):
            scene = self.dataset.get_item(random.choice(list(range(self.dataset.size())))).meta['scene']
        elif isinstance(self.dataset, Iterable_RL_Dataset):
            scene = self.dataset.sample_item().meta['scene']
        else:
            raise NotImplementedError
        self.state = VDObservation(scene)
        return self.state

    def is_terminal(self) -> bool:
        return self.state.event is not None and self.state.event.is_final()

class VDEnvRemoteWrapper:
    def __init__(self, url: str) -> None:
        self.url = url

    def step(self, obs: VDObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, QuestionEvent):
                    history.append({'speaker': 'question', 'text': item.question})
                elif isinstance(item, AnswerEvent):
                    history.append({'speaker': 'answer', 'text': item.answer})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history), 
                   'caption': obs.scene.caption, 
                   'img_features': json.dumps(obs.scene.img_feat.tolist()), 
                   'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}
        a_response, reward = json.loads(requests.post(self.url, 
                                                      data=payload).text)
        return a_response, reward

class VDRemotePolicy(Policy):
    def __init__(self, url: str) -> None:
        self.url = url

    def act(self, obs: VDObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, QuestionEvent):
                    history.append({'speaker': 'question', 'text': item.question})
                elif isinstance(item, AnswerEvent):
                    history.append({'speaker': 'answer', 'text': item.answer})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history), 
                   'caption': obs.scene.caption, 
                   'generation_kwargs': json.dumps({'inference': 'greedy', 'beamSize': 1})}
        q_response = json.loads(requests.post(self.url, 
                                              data=payload).text)
        return q_response

class VDRemoteReward:
    def __init__(self, url: str) -> None:
        self.url = url

    def reward(self, obs: VDObservation):
        history = []
        if obs.event is not None:
            for item in obs.event.get_events():
                if isinstance(item, QuestionEvent):
                    history.append({'speaker': 'question', 'text': item.question})
                elif isinstance(item, AnswerEvent):
                    history.append({'speaker': 'answer', 'text': item.answer})
                else:
                    raise NotImplementedError
        payload = {'history': json.dumps(history), 
                   'caption': obs.scene.caption, 
                   'img_features': json.dumps(obs.scene.img_feat.tolist())}
        q_response = json.loads(requests.post(self.url, 
                                              data=payload).text)
        return q_response
