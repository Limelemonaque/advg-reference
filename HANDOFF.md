# Suggested collaborator note

Dear Professor,

I have prepared a standalone neural reference for AdvG. It provides one training
recipe, used unchanged in the LQ and stochastic Pendulum examples. The actor
and both critics are trainable neural networks. You can run either example with
the commands in the README; both read the same algorithm settings file.

Each training call updates the value critic once, the advantage critic once,
and then the actor once after its scheduled start. AdvG's main
change is to pair each action with the reward and state change observed during
that same interval when training the advantage critic. The implementation note
provides the equations, but there is no need to choose among task-specific
variants to start using the package.

For the diffusion project, we should first define the state, controlled action,
reward and time convention, then run a small neural pilot using the supplied
settings. That will give us a concrete basis for diagnosing or tuning the
application. The reference has been checked on LQ and Pendulum; diffusion-model
performance still needs that pilot.

Best regards,
[Your name]
