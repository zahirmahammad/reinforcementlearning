.. RLDocs documentation master file, created by
   sphinx-quickstart on Mon Mar  2 23:54:23 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reinforcement Learning documentation
====================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.


I tried to keep documentation as concise as possible 
while explaining concepts clearly

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Reinforcement Learning

   mdfiles/01_intro_rl1
   mdfiles/02_intro_rl2
   mdfiles/03_value_based
   mdfiles/04_policy_based

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Value-Based Algorithms

   mdfiles/05_QLearning
   mdfiles/06_DQN

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Policy-Based Algorithms

   mdfiles/07_reinforce
   mdfiles/08_actor_critic
   mdfiles/09_PPO


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Implementation in PyTorch

   mdfiles/110_qlearn
   mdfiles/111_dqn
   mdfiles/112_reinforce
   mdfiles/113_a2c
   mdfiles/114_ppo

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Miscellaneous

   mdfiles/10_multi_agent