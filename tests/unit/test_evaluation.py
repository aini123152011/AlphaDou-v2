"""评估层测试"""
import pytest
import numpy as np


class TestEvalResult:
    """EvalResult 测试"""

    def test_create(self):
        from evaluation import EvalResult

        result = EvalResult(
            win_rate=0.6,
            avg_reward=1.5,
            avg_length=50.0,
            games_played=100,
        )
        assert result.win_rate == 0.6
        assert result.games_played == 100


class TestRandomAgent:
    """RandomAgent 测试"""

    def test_act(self):
        from evaluation import RandomAgent

        agent = RandomAgent()
        obs = {"hand": np.zeros(54)}
        legal_actions = [0, 1, 2, 3, 4]

        action = agent.act(obs, legal_actions)
        assert action in legal_actions

    def test_empty_legal_actions(self):
        from evaluation import RandomAgent

        agent = RandomAgent()
        action = agent.act({}, [])
        assert action == 0


class TestRuleBasedAgent:
    """RuleBasedAgent 测试"""

    def test_prefers_non_pass(self):
        from evaluation import RuleBasedAgent

        agent = RuleBasedAgent()
        obs = {"hand": np.zeros(54)}
        legal_actions = [0, 1, 2, 3]  # 0 是 pass

        action = agent.act(obs, legal_actions)
        assert action != 0  # 应该选择非 pass 动作


class TestEvaluator:
    """Evaluator 测试"""

    def test_evaluate(self):
        from evaluation import Evaluator, RandomAgent
        from env import DoudizhuEnv

        evaluator = Evaluator(env_fn=DoudizhuEnv)
        agent = RandomAgent("test")

        result = evaluator.evaluate(agent, n_games=5)

        assert result.games_played == 5
        assert 0.0 <= result.win_rate <= 1.0

    def test_compare(self):
        from evaluation import Evaluator, RandomAgent
        from env import DoudizhuEnv

        evaluator = Evaluator(env_fn=DoudizhuEnv)
        agent1 = RandomAgent("agent1")
        agent2 = RandomAgent("agent2")

        result = evaluator.compare(agent1, agent2, n_games=5)

        assert "agent1_wins" in result
        assert "agent2_wins" in result
        assert result["agent1_wins"] + result["agent2_wins"] == 5


class TestArena:
    """Arena 测试"""

    def test_play_match(self):
        from evaluation import Arena, RandomAgent
        from env import DoudizhuEnv

        arena = Arena(env_fn=DoudizhuEnv)
        agents = [RandomAgent(f"agent{i}") for i in range(3)]

        results = arena.play_match(agents, n_games=3)

        assert len(results) == 3
        for result in results:
            assert result.winner in ["landlord", "farmer", "draw"]

    def test_round_robin(self):
        from evaluation import Arena, RandomAgent
        from env import DoudizhuEnv

        arena = Arena(env_fn=DoudizhuEnv)
        agents = [RandomAgent(f"agent{i}") for i in range(3)]

        result = arena.round_robin(agents, games_per_match=1)

        assert result.total_games > 0
        for name in ["agent0", "agent1", "agent2"]:
            assert name in result.standings

    def test_tournament(self):
        from evaluation import Arena, RandomAgent
        from env import DoudizhuEnv

        arena = Arena(env_fn=DoudizhuEnv)
        agents = [RandomAgent(f"agent{i}") for i in range(3)]

        result = arena.tournament(agents, n_rounds=5)

        assert result.total_games == 5


class TestLeaderBoard:
    """LeaderBoard 测试"""

    def test_update(self):
        from evaluation import LeaderBoard, TournamentResult

        board = LeaderBoard()

        result = TournamentResult(
            standings={
                "agent1": {"games": 10, "wins": 7},
                "agent2": {"games": 10, "wins": 3},
            },
            total_games=10,
            matches=[],
        )

        board.update(result)

        ranking = board.get_ranking()
        assert len(ranking) == 2


class TestMetricsCollector:
    """MetricsCollector 测试"""

    def test_add_game(self):
        from evaluation import MetricsCollector, GameMetrics

        collector = MetricsCollector()

        metrics = GameMetrics(
            winner="landlord",
            landlord="player1",
            farmers=("player2", "player3"),
            length=50,
            bombs=1,
            spring=False,
        )

        collector.add_game(metrics)

        assert len(collector.games) == 1

    def test_compute_metrics(self):
        from evaluation import MetricsCollector, GameMetrics

        collector = MetricsCollector()

        for i in range(10):
            metrics = GameMetrics(
                winner="landlord" if i < 6 else "farmer",
                landlord="player1",
                farmers=("player2", "player3"),
                length=50 + i,
                bombs=i % 2,
                spring=i == 0,
            )
            collector.add_game(metrics)

        global_metrics = collector.compute_metrics()
        assert global_metrics["total_games"] == 10
        assert global_metrics["landlord_win_rate"] == 0.6

        player_metrics = collector.compute_metrics("player1")
        assert player_metrics["games"] == 10


class TestRunningStats:
    """RunningStats 测试"""

    def test_update(self):
        from evaluation import RunningStats

        stats = RunningStats()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            stats.update(v)

        assert stats.n == 5
        assert abs(stats.mean - 3.0) < 0.01
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0


class TestMetricsAggregator:
    """MetricsAggregator 测试"""

    def test_add_and_get(self):
        from evaluation import MetricsAggregator

        agg = MetricsAggregator()

        for i in range(10):
            agg.add("loss", float(i))
            agg.add("reward", float(i) * 2)

        loss_stats = agg.get("loss")
        assert loss_stats["count"] == 10
        assert abs(loss_stats["mean"] - 4.5) < 0.01

        means = agg.get_means()
        assert "loss" in means
        assert "reward" in means


class TestComputeMetrics:
    """指标计算测试"""

    def test_action_entropy(self):
        from evaluation import compute_action_entropy

        # 均匀分布应该有最大熵
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_action_entropy(uniform)
        assert entropy > 1.0

        # 确定性分布应该有最小熵
        deterministic = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_action_entropy(deterministic)
        assert entropy < 0.01

    def test_value_accuracy(self):
        from evaluation import compute_value_accuracy

        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        actual = np.array([1.05, 2.05, 3.05, 4.05])

        accuracy = compute_value_accuracy(predicted, actual, threshold=0.1)
        assert accuracy == 1.0

    def test_explained_variance(self):
        from evaluation import compute_explained_variance

        # 完美预测
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ev = compute_explained_variance(predicted, actual)
        assert abs(ev - 1.0) < 0.01


class TestEloSystem:
    """EloSystem 测试"""

    def test_initial_rating(self):
        from evaluation import EloSystem

        elo = EloSystem()
        player = elo.get_player("test")

        assert player.rating == 1500.0
        assert player.games == 0

    def test_record_match(self):
        from evaluation import EloSystem

        elo = EloSystem()

        winner_rating, loser_rating = elo.record_match("winner", "loser")

        assert winner_rating > 1500.0
        assert loser_rating < 1500.0

    def test_expected_score(self):
        from evaluation import EloSystem

        elo = EloSystem()

        # 相同评分应该是 0.5
        expected = elo.expected_score(1500.0, 1500.0)
        assert abs(expected - 0.5) < 0.01

        # 评分高的玩家期望得分更高
        expected_high = elo.expected_score(1600.0, 1400.0)
        assert expected_high > 0.5

    def test_get_ranking(self):
        from evaluation import EloSystem

        elo = EloSystem()

        elo.record_match("winner1", "loser1")
        elo.record_match("winner1", "loser2")
        elo.record_match("winner2", "loser1")

        ranking = elo.get_ranking()
        assert len(ranking) == 4
        assert ranking[0].rating > ranking[-1].rating


class TestMultiPlayerElo:
    """MultiPlayerElo 测试"""

    def test_record_game(self):
        from evaluation import MultiPlayerElo

        elo = MultiPlayerElo()

        elo.record_game(
            landlord="landlord1",
            farmers=("farmer1", "farmer2"),
            winner="landlord",
        )

        landlord = elo.get_player("landlord1")
        farmer1 = elo.get_player("farmer1")

        assert landlord.rating > 1500.0
        assert farmer1.rating < 1500.0


class TestTrueSkillLite:
    """TrueSkillLite 测试"""

    def test_initial_rating(self):
        from evaluation import TrueSkillLite

        ts = TrueSkillLite()
        mu, sigma = ts.get_player("test")

        assert mu == 25.0
        assert abs(sigma - 25.0 / 3.0) < 0.01

    def test_update(self):
        from evaluation import TrueSkillLite

        ts = TrueSkillLite()

        ts.update(winners=["winner"], losers=["loser"])

        winner_mu, _ = ts.get_player("winner")
        loser_mu, _ = ts.get_player("loser")

        assert winner_mu > 25.0
        assert loser_mu < 25.0

    def test_win_probability(self):
        from evaluation import TrueSkillLite

        ts = TrueSkillLite()

        # 相同技能应该接近 0.5
        prob = ts.win_probability(["player1"], ["player2"])
        assert abs(prob - 0.5) < 0.1


class TestGlickoSystem:
    """GlickoSystem 测试"""

    def test_initial_rating(self):
        from evaluation import GlickoSystem

        glicko = GlickoSystem()
        rating, rd = glicko.get_player("test")

        assert rating == 1500.0
        assert rd == 350.0

    def test_update(self):
        from evaluation import GlickoSystem

        glicko = GlickoSystem()

        glicko.update("player1", [("player2", 1.0)])  # player1 赢了 player2

        rating1, _ = glicko.get_player("player1")
        assert rating1 > 1500.0
