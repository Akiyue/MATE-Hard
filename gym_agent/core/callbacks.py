class Callbacks:
    def __init__(self, agent) -> None:
        self.agent = agent
        
    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_episode_begin(self, **kwargs):
        pass

    def on_episode_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass

    def on_action_begin(self, **kwargs):
        pass

    def on_action_end(self, **kwargs):
        pass

    def on_learn_begin(self, **kwargs):
        pass

    def on_learn_end(self, **kwargs):
        pass

    def on_reset(self, **kwargs):
        pass

    def on_play_begin(self, **kwargs):
        pass

    def on_play_end(self, **kwargs):
        pass
