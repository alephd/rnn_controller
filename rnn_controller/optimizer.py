from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GenericOptimizer(object):

    def __init__(self, initializer, name, learning_rate, **kwargs):
        # TODO(nperrin16): Add kwargs in name?
        self.name = f"{name}_{learning_rate}"
        self.learning_rate = learning_rate
        self.initializer = lambda learning_rate, global_step: initializer(learning_rate=learning_rate, name=name, **kwargs)
        self.learning_rate_schedule = lambda global_step: self.learning_rate

    def to_model_params(self):
        return {"optimizer_name": self.name,
                "optimizer_initializer": self.initializer,
                "learning_rate_schedule": self.learning_rate_schedule
                }


class ScheduledSGDOptimizer(GenericOptimizer):

    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase,
                 decay_fn=tf.train.inverse_time_decay):
        if decay_fn == tf.train.inverse_time_decay:
            decay_fn_name = 'itDK'
        elif decay_fn == tf.train.exponential_decay:
            decay_fn_name = 'expDK'
        else:
            raise ValueError("Only tf.train.inverse_time_decay or tf.train.exponential_decay are allowed")
        super().__init__(initializer=tf.train.GradientDescentOptimizer, name=f"sgd_{decay_fn_name}",
                         learning_rate=initial_learning_rate)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.decay_fn = decay_fn
        # Override name and learning_rate_schedule
        self.name = f"{self.name}_{decay_steps}_{decay_rate}_{str(staircase)[0]}"
        self.learning_rate_schedule = lambda gs: self.decay_fn(self.learning_rate, gs, decay_steps=self.decay_steps,
                                                               decay_rate=self.decay_rate, staircase=self.staircase)


class AnnealedSGDOptimizer(GenericOptimizer):

    def __init__(self, initial_learning_rate, decay_steps, t_mul=2, m_mul=1.0, alpha=0.0,
                 decay_fn=tf.train.cosine_decay_restarts):
        if decay_fn == tf.train.cosine_decay_restarts:
            decay_fn_name = 'cosWRDK'
        else:
            raise ValueError("Only tf.train.cosine_decay_restarts allowed")
        super().__init__(initializer=tf.train.GradientDescentOptimizer, name=f"sgd_{decay_fn_name}",
                         learning_rate=initial_learning_rate)
        self.decay_steps = decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.decay_fn = decay_fn
        # Override name and learning_rate_schedule
        if decay_steps == 50:  # Quick fix for legacy runs to be reusable # TODO(at): remove
            self.name = f"{self.name}_{t_mul}_{m_mul}_{alpha}"
        else:
            self.name = f"{self.name}_{decay_steps}_{t_mul}_{m_mul}_{alpha}"
        self.learning_rate_schedule = lambda gs: self.decay_fn(self.learning_rate, gs,
                                                               first_decay_steps=self.decay_steps,
                                                               t_mul=self.t_mul, m_mul=self.m_mul, alpha=self.alpha)


DEFAULT_OPTIMIZER_PARAMS = {f"{n}_{l}": GenericOptimizer(initializer=i, name=n, learning_rate=l).to_model_params()
                            for i, n, l in [(tf.train.AdamOptimizer, "adam", 0.1),
                                            (tf.train.RMSPropOptimizer, "rms", 0.1),
                                            (tf.train.RMSPropOptimizer, "rms", 0.01),
                                            (tf.train.RMSPropOptimizer, "rms", 0.001),
                                            (tf.train.RMSPropOptimizer, "rms", 1e-4), ]}
