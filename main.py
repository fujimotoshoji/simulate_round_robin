import polars as pl
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """シミュレーション設定"""

    player_strengths: Dict[str, float]  # 参加者の実力値 {プレイヤー名: 実力値}
    num_simulations: int = 10000  # シミュレーション回数
    num_matches: int = 1  # 各ペアの対戦回数
    random_seed: int = 42  # 乱数シード


class RoundRobinAnalyzer:
    """総当たり戦順位決定アルゴリズム検証クラス（Polars版）"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.players = list(config.player_strengths.keys())
        self.num_players = len(self.players)
        self.player_to_idx = {player: idx for idx, player in enumerate(self.players)}
        self.idx_to_player = {idx: player for player, idx in self.player_to_idx.items()}

        np.random.seed(config.random_seed)

        # 真の実力順（降順）
        sorted_players = sorted(
            config.player_strengths.items(), key=lambda x: x[1], reverse=True
        )
        self.true_ranking = [player for player, _ in sorted_players]
        logger.info(f"真の実力順: {self.true_ranking}")

        # 対戦ペアの生成
        self.match_pairs = self._generate_match_pairs()

    def _generate_match_pairs(self) -> pl.DataFrame:
        """対戦ペアのDataFrameを生成"""
        pairs = []
        for i, player_i in enumerate(self.players):
            for j, player_j in enumerate(self.players[i + 1 :], i + 1):
                pairs.append(
                    {
                        "player_i": player_i,
                        "player_j": player_j,
                        "player_i_idx": i,
                        "player_j_idx": j,
                        "strength_i": self.config.player_strengths[player_i],
                        "strength_j": self.config.player_strengths[player_j],
                    }
                )

        return pl.DataFrame(pairs)

    def calculate_win_probability(self, strength_i: float, strength_j: float) -> float:
        """イロレーティングベースの勝率計算"""
        return 1 / (1 + 10 ** ((strength_j - strength_i) / 400))

    def create_simulation_table(self) -> pl.DataFrame:
        """シミュレーション用のテーブルを作成"""
        # シミュレーション番号のリストを作成
        simulation_numbers = pl.DataFrame(
            {"simulation": range(self.config.num_simulations)}
        )

        # クロス結合で全シミュレーション × 全対戦ペアのテーブルを作成
        # cross_joinの代わりにjoinを使用
        simulation_table = simulation_numbers.join(self.match_pairs, how="cross")

        # 勝率を計算
        simulation_table = simulation_table.with_columns(
            [
                (
                    1
                    / (
                        1
                        + pl.lit(10)
                        ** ((pl.col("strength_j") - pl.col("strength_i")) / 400)
                    )
                ).alias("win_prob")
            ]
        )

        return simulation_table

    def add_random_values(self, simulation_table: pl.DataFrame) -> pl.DataFrame:
        """乱数値を追加"""
        # 各対戦に一意の乱数を生成
        total_matches = len(simulation_table)
        random_values = np.random.random(total_matches)

        # 乱数値を追加
        simulation_table = simulation_table.with_columns(
            [pl.lit(random_values).alias("random_value")]
        )

        return simulation_table

    def determine_match_results(self, simulation_table: pl.DataFrame) -> pl.DataFrame:
        """勝敗を判定"""
        # 勝敗判定: random_value < win_prob なら player_i勝利
        match_results = simulation_table.with_columns(
            [
                (pl.col("random_value") < pl.col("win_prob"))
                .cast(pl.Int8)
                .alias("result")
            ]
        )

        return match_results

    def calculate_standings(self, match_results: pl.DataFrame) -> pl.DataFrame:
        """順位表を計算"""
        # 各プレイヤーの勝ち数を集計
        # player_iとしての勝ち数（result == 1の時）
        wins_as_i = (
            match_results.filter(pl.col("result") == 1)
            .group_by(["simulation", "player_i"])
            .agg([pl.len().alias("wins_as_i")])
            .rename({"player_i": "player"})
        )

        # player_jとしての勝ち数（result == 0の時）
        wins_as_j = (
            match_results.filter(pl.col("result") == 0)
            .group_by(["simulation", "player_j"])
            .agg([pl.len().alias("wins_as_j")])
            .rename({"player_j": "player"})
        )

        # 全プレイヤーを含むベースDataFrameを作成
        all_players = []
        for sim in range(self.config.num_simulations):
            for player in self.players:
                all_players.append({"simulation": sim, "player": player})

        base_df = pl.DataFrame(all_players)

        # 勝ち数を結合して総勝ち数を計算
        standings = (
            base_df.join(wins_as_i, on=["simulation", "player"], how="left")
            .join(wins_as_j, on=["simulation", "player"], how="left")
            .fill_null(0)
            .with_columns([(pl.col("wins_as_i") + pl.col("wins_as_j")).alias("wins")])
            .select(["simulation", "player", "wins"])
        )

        # 負け数は総対戦数 - 勝ち数で計算
        total_matches_per_player = self.num_players - 1
        standings = standings.with_columns(
            [(pl.lit(total_matches_per_player) - pl.col("wins")).alias("losses")]
        )

        return standings

    def find_tied_players(self, standings: pl.DataFrame) -> pl.DataFrame:
        """同順位群を検出（2人のみ対象）"""
        tied_groups = (
            standings.group_by(["simulation", "wins", "losses"])
            .agg([pl.col("player").alias("tied_players")])
            .filter((pl.col("tied_players").list.len() == 2))  # 2人のみ対象
        )

        return tied_groups

    def determine_ranking_by_direct_match(
        self, tied_group: List[str], match_results: pl.DataFrame, sim: int
    ) -> List[str]:
        """直接対決結果による順位決定（従来ルール）"""
        if len(tied_group) != 2:  # 2人のみ対象
            return tied_group

        # 2人の直接対決結果を確認
        relevant_matches = match_results.filter(
            (pl.col("simulation") == sim)
            & (pl.col("player_i").is_in(tied_group))
            & (pl.col("player_j").is_in(tied_group))
        )

        if len(relevant_matches) == 0:
            return tied_group  # 直接対決がない場合は元の順序

        # 直接対決の結果で順位決定
        row = relevant_matches.iter_rows(named=True).__next__()
        if row["result"] == 1:
            return [row["player_i"], row["player_j"]]  # player_i勝利
        else:
            return [row["player_j"], row["player_i"]]  # player_j勝利

    def determine_ranking_by_direct_match_reverse(
        self, tied_group: List[str], match_results: pl.DataFrame, sim: int
    ) -> List[str]:
        """直接対決結果による順位決定（代替ルール：敗者が上位）"""
        normal_ranking = self.determine_ranking_by_direct_match(
            tied_group, match_results, sim
        )
        return normal_ranking[::-1]

    def check_ranking_accuracy(
        self, determined_ranking: List[str], true_ranking: List[str]
    ) -> bool:
        """決定された順位が真の実力順と一致するかチェック"""
        if len(determined_ranking) != len(true_ranking):
            return False

        true_sub_ranking = [p for p in true_ranking if p in determined_ranking]
        return determined_ranking == true_sub_ranking

    def run_simulation(self) -> Dict:
        """モンテカルロシミュレーションの実行"""
        logger.info(f"シミュレーション開始: {self.config.num_simulations}回")

        # 1. シミュレーションテーブルを作成
        simulation_table = self.create_simulation_table()
        logger.info("シミュレーションテーブル作成完了")

        # 2. 乱数値を追加
        simulation_table = self.add_random_values(simulation_table)
        logger.info("乱数値追加完了")

        # 3. 勝敗を判定
        match_results = self.determine_match_results(simulation_table)
        logger.info("勝敗判定完了")

        # 勝敗テーブルの中身を確認
        self.print_match_results_sample(match_results)

        # 4. 順位表を計算
        standings = self.calculate_standings(match_results)
        logger.info("順位表計算完了")

        # 5. 同順位群を検出
        tied_groups = self.find_tied_players(standings)
        logger.info(f"同順位群検出完了: {len(tied_groups)}件")

        # 結果集計
        results = {
            "tie_occurrences": len(tied_groups),
            "correct_rankings_normal": 0,
            "correct_rankings_reverse": 0,
            "tied_groups_data": [],
            "win_probability_matrix": self._calculate_win_probability_matrix(),
            "match_results": match_results,  # 勝敗テーブルを保存
        }

        # 各同順位群での順位決定精度を計算
        for row in tied_groups.iter_rows(named=True):
            sim = row["simulation"]
            tied_group = row["tied_players"]

            true_sub_ranking = [p for p in self.true_ranking if p in tied_group]

            normal_ranking = self.determine_ranking_by_direct_match(
                tied_group, match_results, sim
            )
            if self.check_ranking_accuracy(normal_ranking, true_sub_ranking):
                results["correct_rankings_normal"] += 1

            reverse_ranking = self.determine_ranking_by_direct_match_reverse(
                tied_group, match_results, sim
            )
            if self.check_ranking_accuracy(reverse_ranking, true_sub_ranking):
                results["correct_rankings_reverse"] += 1

            results["tied_groups_data"].append(
                {
                    "simulation": sim,
                    "tied_group": tied_group,
                    "true_ranking": true_sub_ranking,
                    "normal_ranking": normal_ranking,
                    "reverse_ranking": reverse_ranking,
                }
            )

        return results

    def print_match_results_sample(self, match_results: pl.DataFrame):
        """勝敗テーブルのサンプルを表示"""
        print("\n" + "=" * 80)
        print("勝敗テーブル サンプル（最初の20行）")
        print("=" * 80)

        sample = match_results.head(20)
        print(sample)

        # 統計情報も表示
        print(f"\n総対戦数: {len(match_results):,}")
        print(f"player_i勝利数: {match_results.filter(pl.col('result') == 1).height:,}")
        print(f"player_j勝利数: {match_results.filter(pl.col('result') == 0).height:,}")

        # 勝率の分布も確認
        print("\n勝率の統計:")
        win_prob_stats = match_results.select(pl.col("win_prob")).describe()
        print(win_prob_stats)

        # 最初の数戦の詳細結果を表示
        print("\n" + "=" * 80)
        print("最初の数戦の詳細結果")
        print("=" * 80)

        # 最初の5回のシミュレーションの各対戦結果を表示
        for sim in range(min(5, self.config.num_simulations)):
            print(f"\n=== シミュレーション {sim} ===")
            sim_matches = match_results.filter(pl.col("simulation") == sim)

            for row in sim_matches.iter_rows(named=True):
                winner = row["player_i"] if row["result"] == 1 else row["player_j"]
                loser = row["player_j"] if row["result"] == 1 else row["player_i"]
                win_prob = row["win_prob"]
                random_val = row["random_value"]

                print(
                    f"  {row['player_i']} vs {row['player_j']}: "
                    f"{winner} 勝利 (勝率: {win_prob:.3f}, 乱数: {random_val:.3f})"
                )

        # 最初の5回のシミュレーションの順位表も表示
        print("\n" + "=" * 80)
        print("最初の5回のシミュレーションの順位表")
        print("=" * 80)

        standings = self.calculate_standings(match_results)

        for sim in range(min(5, self.config.num_simulations)):
            print(f"\n--- シミュレーション {sim} ---")
            sim_standings = standings.filter(pl.col("simulation") == sim).sort(
                ["wins", "losses"], descending=[True, False]
            )

            for i, row in enumerate(sim_standings.iter_rows(named=True)):
                print(f"  {i+1}位: {row['player']} ({row['wins']}勝 {row['losses']}敗)")

    def _calculate_win_probability_matrix(self) -> np.ndarray:
        """勝率行列の計算"""
        matrix = np.zeros((self.num_players, self.num_players))
        for i, player_i in enumerate(self.players):
            for j, player_j in enumerate(self.players):
                if i != j:
                    matrix[i, j] = self.calculate_win_probability(
                        self.config.player_strengths[player_i],
                        self.config.player_strengths[player_j],
                    )
        return matrix

    def print_summary(self, results: Dict):
        """シミュレーション結果のサマリを出力"""
        print("\n" + "=" * 60)
        print("シミュレーションサマリ")
        print("=" * 60)
        print(f"参加者: {list(self.config.player_strengths.keys())}")
        print(f"実力値: {list(self.config.player_strengths.values())}")
        print(f"真の実力順: {self.true_ranking}")
        print(f"総シミュレーション回数: {self.config.num_simulations:,}")
        print(f"同順位発生回数: {results['tie_occurrences']:,}")
        print(
            f"同順位発生率: {results['tie_occurrences']/self.config.num_simulations*100:.2f}%"
        )

        if results["tie_occurrences"] > 0:
            print(f"\n従来ルール（直接対決勝者が上位）:")
            print(f"  正解回数: {results['correct_rankings_normal']:,}")
            print(
                f"  精度: {results['correct_rankings_normal']/results['tie_occurrences']*100:.2f}%"
            )

            print(f"\n代替ルール（直接対決敗者が上位）:")
            print(f"  正解回数: {results['correct_rankings_reverse']:,}")
            print(
                f"  精度: {results['correct_rankings_reverse']/results['tie_occurrences']*100:.2f}%"
            )

    def print_win_probability_table(self, results: Dict):
        """勝率表の出力"""
        print("\n" + "=" * 60)
        print("勝率表")
        print("=" * 60)

        matrix = results["win_probability_matrix"]

        win_prob_df = pl.DataFrame({"player": self.players})

        for j, player_j in enumerate(self.players):
            win_prob_df = win_prob_df.with_columns(
                [pl.lit(matrix[:, j]).alias(f"vs_{player_j}")]
            )

        print(win_prob_df)

    def create_detailed_analysis(self, results: Dict):
        """詳細分析の作成"""
        if not results["tied_groups_data"]:
            print("詳細分析するデータがありません（同順位が発生しませんでした）")
            return

        tied_data = []
        for data in results["tied_groups_data"]:
            tied_data.append(
                {
                    "simulation": data["simulation"],
                    "group_size": len(data["tied_group"]),
                    "tied_players": str(data["tied_group"]),
                    "true_ranking": str(data["true_ranking"]),
                    "normal_ranking": str(data["normal_ranking"]),
                    "reverse_ranking": str(data["reverse_ranking"]),
                    "normal_correct": data["normal_ranking"] == data["true_ranking"],
                    "reverse_correct": data["reverse_ranking"] == data["true_ranking"],
                }
            )

        tied_df = pl.DataFrame(tied_data)

        print("\n" + "=" * 60)
        print("詳細分析（2人同順位のみ対象）")
        print("=" * 60)

        # 2人同順位の分析
        size_analysis = tied_df.group_by("group_size").agg(
            [
                pl.count().alias("count"),
                pl.col("normal_correct").mean().alias("normal_accuracy"),
                pl.col("reverse_correct").mean().alias("reverse_accuracy"),
            ]
        )

        print("同順位群サイズ別の精度:")
        print(size_analysis)

        # 3人以上の同順位が発生した場合は報告
        large_groups = tied_df.filter(pl.col("group_size") >= 3)
        if len(large_groups) > 0:
            print(
                f"\n注意: 3人以上の同順位が{len(large_groups)}件発生しました（集計対象外）"
            )
            print("最初の5件:")
            for row in large_groups.head(5).iter_rows(named=True):
                print(f"  シミュレーション{row['simulation']}: {row['tied_players']}")

    def create_visualizations(self, results: Dict):
        """結果の可視化"""
        if not results["tied_groups_data"]:
            print("可視化するデータがありません（同順位が発生しませんでした）")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 英語のタイトルに変更
        rules = ["Traditional Rule", "Alternative Rule"]
        accuracies = [
            results["correct_rankings_normal"] / results["tie_occurrences"] * 100,
            results["correct_rankings_reverse"] / results["tie_occurrences"] * 100,
        ]

        bars = ax1.bar(rules, accuracies, color=["skyblue", "lightcoral"])
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Ranking Rule Accuracy Comparison")
        ax1.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
            )

        im = ax2.imshow(
            results["win_probability_matrix"], cmap="RdYlBu_r", vmin=0, vmax=1
        )
        ax2.set_xticks(range(self.num_players))
        ax2.set_yticks(range(self.num_players))
        ax2.set_xticklabels(self.players)
        ax2.set_yticklabels(self.players)
        ax2.set_title("Win Probability Matrix")

        plt.colorbar(im, ax=ax2, label="Win Probability")

        plt.tight_layout()
        plt.savefig("simulation_results.png", dpi=300, bbox_inches="tight")

        # 非インタラクティブ環境ではshow()をスキップ
        try:
            plt.show()
        except:
            print("グラフをsimulation_results.pngに保存しました")


def main():
    """メイン関数"""
    # イロレーティングベースの設定
    config = SimulationConfig(
        player_strengths={
            "Alice": 2400,  # 強いプレイヤー
            "Bob": 2300,  # 200ポイント差
            "Charlie": 2200,  # 200ポイント差
            "Diana": 2100,  # 200ポイント差
            "Eve": 2000,  # 200ポイント差
        },
        num_simulations=100000,
        num_matches=1,
        random_seed=42,
    )

    # アナライザーの初期化
    analyzer = RoundRobinAnalyzer(config)

    # シミュレーション実行
    results = analyzer.run_simulation()

    # 結果出力
    analyzer.print_summary(results)
    analyzer.print_win_probability_table(results)

    # 詳細分析
    analyzer.create_detailed_analysis(results)

    # 可視化
    analyzer.create_visualizations(results)

    print("\nシミュレーション完了！")


if __name__ == "__main__":
    main()
