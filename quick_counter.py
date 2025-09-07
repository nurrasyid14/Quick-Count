import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import statsmodels.api as sm
import scipy.stats as stats
import os

class QuickCountAnalyzer:
    """Performs statistical analysis on election quick count data from Nganjuk"""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the analyzer with data from CSV file
        
        Args:
            csv_file_path: Path to the CSV file containing election data
        """
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
        
        # Load and preprocess the data
        self.data = pd.read_csv(csv_file_path)
        
        # Rename columns for consistency
        column_mapping = {
            'kecamatan': 'District',
            'kelurahan': 'Village',
            'paslon_1': 'O1',
            'paslon_2': 'O2',
            'paslon_3': 'O3',
            'total_suara_sah': 'Sum_of_Voices',
            'suara_tidak_sah': 'Invalid_Votes'
        }
        
        # Only rename columns that exist in the dataframe
        existing_columns = {k: v for k, v in column_mapping.items() if k in self.data.columns}
        self.data.rename(columns=existing_columns, inplace=True)
        
        # Identify candidate columns dynamically
        self.candidates = [col for col in ['O1', 'O2', 'O3'] if col in self.data.columns]
        
        # Initialize sampler
        self.sampler = Sampler(self.data)
    
    def calculate_totals(self) -> Dict[str, int]:
        """Calculate total votes for each candidate"""
        totals = {}
        for candidate in self.candidates:
            totals[candidate] = self.data[candidate].sum()
        return totals
    
    def calculate_percentages(self) -> Dict[str, float]:
        """Calculate percentage of votes for each candidate"""
        totals = self.calculate_totals()
        total_valid_votes = self.data['Sum_of_Voices'].sum()
        
        percentages = {}
        for candidate, votes in totals.items():
            percentages[candidate] = (votes / total_valid_votes) * 100
            
        return percentages
    
    def calculate_voter_turnout(self, eligible_voters: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate voter turnout statistics
        
        Args:
            eligible_voters: If provided, use this value instead of estimating
        """
        total_valid_votes = self.data['Sum_of_Voices'].sum()
        total_invalid_votes = self.data['Invalid_Votes'].sum()
        total_votes_cast = total_valid_votes + total_invalid_votes
        
        if eligible_voters is not None:
            voter_turnout_percentage = (total_votes_cast / eligible_voters) * 100
            estimated_eligible_voters = eligible_voters
        else:
            # Estimate eligible voters based on average Indonesian turnout (~75%)
            estimated_eligible_voters = total_votes_cast / 0.75
            voter_turnout_percentage = 75.0  # Default estimate
        
        return {
            'valid_votes': total_valid_votes,
            'invalid_votes': total_invalid_votes,
            'total_votes_cast': total_votes_cast,
            'voter_turnout_percentage': voter_turnout_percentage,
            'estimated_eligible_voters': estimated_eligible_voters
        }
    
    def calculate_by_district(self) -> pd.DataFrame:
        """Calculate results by district"""
        if 'District' not in self.data.columns:
            raise ValueError("District column not found in the data")
            
        district_results = self.data.groupby('District').agg({
            'O1': 'sum',
            'O2': 'sum',
            'O3': 'sum',
            'Sum_of_Voices': 'sum',
            'Invalid_Votes': 'sum'
        }).reset_index()
        
        for candidate in self.candidates:
            district_results[f'{candidate}_pct'] = (district_results[candidate] / 
                                                   district_results['Sum_of_Voices']) * 100
                                                   
        return district_results
    
    def calculate_margin_of_error(self, confidence_level=0.95) -> Dict[str, float]:
        """
        Calculate margin of error for each candidate
        Using binomial distribution approximation
        """
        percentages = self.calculate_percentages()
        total_votes = self.data['Sum_of_Voices'].sum()
        
        # Z-score for confidence level
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.99:
            z = 2.576
        else:
            # Calculate z-score for other confidence levels
            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        margins = {}
        for candidate, pct in percentages.items():
            p = pct / 100  # Convert to proportion
            moe = z * np.sqrt((p * (1 - p)) / total_votes) * 100
            margins[candidate] = moe
            
        return margins
    
    def calculate_confidence_intervals(self, confidence_level=0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for each candidate"""
        percentages = self.calculate_percentages()
        margins = self.calculate_margin_of_error(confidence_level)
        
        intervals = {}
        for candidate in self.candidates:
            pct = percentages[candidate]
            moe = margins[candidate]
            intervals[candidate] = (max(0, pct - moe), min(100, pct + moe))
            
        return intervals
    
    def identify_strongholds(self, threshold=0.6) -> Dict[str, List[str]]:
        """
        Identify stronghold districts for each candidate
        A stronghold is defined as a district where a candidate received more than threshold percentage of votes
        """
        if 'District' not in self.data.columns:
            return {candidate: [] for candidate in self.candidates}
            
        district_results = self.calculate_by_district()
        strongholds = {candidate: [] for candidate in self.candidates}
        
        for _, row in district_results.iterrows():
            district = row['District']
            for candidate in self.candidates:
                if row[f'{candidate}_pct'] > threshold * 100:  # Convert to percentage
                    strongholds[candidate].append(district)
                    
        return strongholds
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation between candidate votes across villages"""
        correlation_matrix = self.data[self.candidates].corr()
        return correlation_matrix
    
    def plot_results(self, save_path: Optional[str] = None):
        """Create visualization of election results"""
        # Set up the plot style
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall results
        totals = self.calculate_totals()
        candidates = list(totals.keys())
        votes = list(totals.values())
        
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        bars = ax1.bar(candidates, votes, color=colors)
        ax1.set_title('Total Votes by Candidate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Votes', fontweight='bold')
        
        # Add value labels on bars
        for i, v in enumerate(votes):
            ax1.text(i, v + max(votes)*0.01, f'{v:,}', ha='center', fontweight='bold')
        
        # Plot 2: Results by district
        try:
            district_results = self.calculate_by_district()
            x = np.arange(len(district_results))
            width = 0.25
            
            for i, candidate in enumerate(self.candidates):
                offset = width * i
                ax2.bar(x + offset, district_results[f'{candidate}_pct'], 
                       width, label=candidate, color=colors[i])
            
            ax2.set_xlabel('District', fontweight='bold')
            ax2.set_ylabel('Percentage of Votes', fontweight='bold')
            ax2.set_title('Vote Percentage by District', fontsize=14, fontweight='bold')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels([d for d in district_results['District']], rotation=45, ha='right')
            ax2.legend()
        except ValueError:
            ax2.text(0.5, 0.5, 'District data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Vote Percentage by District (Data Not Available)')
        
        # Plot 3: Voter turnout by district
        try:
            district_turnout = district_results.copy()
            district_turnout['Turnout'] = (district_turnout['Sum_of_Voices'] + district_turnout['Invalid_Votes']) / district_turnout['Sum_of_Voices'].mean() * 100
            
            ax3.bar(district_turnout['District'], district_turnout['Turnout'], color='skyblue')
            ax3.set_xlabel('District', fontweight='bold')
            ax3.set_ylabel('Turnout Index (100 = Average)', fontweight='bold')
            ax3.set_title('Voter Turnout by District', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        except (ValueError, NameError):
            ax3.text(0.5, 0.5, 'District data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Voter Turnout by District (Data Not Available)')
        
        # Plot 4: Correlation heatmap
        correlation_matrix = self.calculate_correlation_matrix()
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_xticks(np.arange(len(self.candidates)))
        ax4.set_yticks(np.arange(len(self.candidates)))
        ax4.set_xticklabels(self.candidates)
        ax4.set_yticklabels(self.candidates)
        ax4.set_title('Correlation Between Candidate Votes', fontsize=14, fontweight='bold')
        
        # Add correlation values to heatmap
        for i in range(len(self.candidates)):
            for j in range(len(self.candidates)):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="w", fontweight='bold')
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print a comprehensive summary of the quick count"""
        print("=" * 60)
        print("REGENT ELECTION QUICK COUNT RESULTS - NGANJUK")
        print("=" * 60)
        
        # Voter turnout
        turnout = self.calculate_voter_turnout()
        print(f"Total Valid Votes: {turnout['valid_votes']:,}")
        print(f"Invalid Votes: {turnout['invalid_votes']:,}")
        print(f"Total Votes Cast: {turnout['total_votes_cast']:,}")
        print(f"Estimated Eligible Voters: {turnout['estimated_eligible_voters']:,.0f}")
        print(f"Estimated Voter Turnout: {turnout['voter_turnout_percentage']:.1f}%")
        print()
        
        # Candidate results
        totals = self.calculate_totals()
        percentages = self.calculate_percentages()
        intervals = self.calculate_confidence_intervals()
        
        print("CANDIDATE PERFORMANCE:")
        print("-" * 50)
        for candidate in self.candidates:
            print(f"{candidate}:")
            print(f"  Votes: {totals[candidate]:,}")
            print(f"  Percentage: {percentages[candidate]:.2f}%")
            print(f"  95% Confidence Interval: {intervals[candidate][0]:.2f}% - {intervals[candidate][1]:.2f}%")
            print()
        
        # Margin of error
        margins = self.calculate_margin_of_error()
        print(f"Overall Margin of Error: ±{margins[self.candidates[0]]:.2f}%")
        print()
        
        # Strongholds
        try:
            strongholds = self.identify_strongholds(threshold=0.55)  # 55% threshold for stronghold
            print("CANDIDATE STRONGHOLDS (>55% of vote):")
            print("-" * 50)
            for candidate, districts in strongholds.items():
                if districts:
                    print(f"{candidate}: {', '.join(districts)}")
                else:
                    print(f"{candidate}: No strongholds identified")
            print()
        except:
            print("Stronghold analysis not available (district data missing)")
            print()
        
        # Potential winner
        sorted_candidates = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_candidates) >= 2:
            first, second = sorted_candidates[0], sorted_candidates[1]
            
            # Check if the difference is statistically significant
            diff = first[1] - second[1]
            avg_moe = (margins[first[0]] + margins[second[0]]) / 2
            
            print("PROJECTION:")
            print("-" * 50)
            if diff > avg_moe * 2:
                print(f"{first[0]} is projected to win with a statistically significant lead")
            elif diff > avg_moe:
                print(f"{first[0]} is leading, but the race is still competitive")
            else:
                print(f"The race is too close to call between {first[0]} and {second[0]}")
        else:
            print("Not enough candidates for projection")
            
        print("=" * 60)
    
    def run_sampling_analysis(self, sample_size: int = 100):
        """Run analysis on a sample of the data to simulate quick count"""
        print("SAMPLING ANALYSIS (Simulated Quick Count)")
        print("-" * 50)
        
        # Take a random sample
        sample_data = self.sampler.random_sample(sample_size)
        
        # Calculate sample statistics
        sample_totals = {candidate: sample_data[candidate].sum() for candidate in self.candidates}
        sample_total_votes = sample_data['Sum_of_Voices'].sum()
        sample_percentages = {candidate: (sample_totals[candidate] / sample_total_votes) * 100 
                             for candidate in self.candidates}
        
        print(f"Sample size: {sample_size} polling stations")
        print("Sample results:")
        for candidate in self.candidates:
            print(f"  {candidate}: {sample_percentages[candidate]:.2f}%")
        
        # Compare with full results
        full_percentages = self.calculate_percentages()
        print("\nComparison with full results:")
        for candidate in self.candidates:
            diff = abs(sample_percentages[candidate] - full_percentages[candidate])
            print(f"  {candidate}: Sample {sample_percentages[candidate]:.2f}% vs "
                  f"Full {full_percentages[candidate]:.2f}% (Difference: {diff:.2f}%)")
        
        # Check if sample would predict the same winner
        sample_winner = max(sample_percentages.items(), key=lambda x: x[1])[0]
        full_winner = max(full_percentages.items(), key=lambda x: x[1])[0]
        
        print(f"\nSample prediction: {sample_winner} would win")
        print(f"Actual result: {full_winner} won")
        print(f"Prediction correct: {sample_winner == full_winner}")



###################################################################################################
###################################################################################################
###################################################################################################



class Sampler:
    def __init__(self, data):
        self.data = data
    
    def random_sample(self, n: int) -> pd.DataFrame:
        return self.data.sample(n, random_state=42)  # Fixed seed for reproducibility
    
    def stratified_sample(self, stratify_by: str, n_per_stratum: int) -> pd.DataFrame:
        if stratify_by not in self.data.columns:
            raise ValueError(f"Column {stratify_by} not found in data")
        return self.data.groupby(stratify_by).apply(lambda x: x.sample(
            min(n_per_stratum, len(x)), random_state=42)).reset_index(drop=True)
    
    def systematic_sample(self, step: int) -> pd.DataFrame:
        return self.data.iloc[::step, :].reset_index(drop=True)
    
    def cluster_sample(self, cluster_by: str, n_clusters: int) -> pd.DataFrame:
        if cluster_by not in self.data.columns:
            raise ValueError(f"Column {cluster_by} not found in data")
        clusters = self.data[cluster_by].unique()
        selected_clusters = np.random.choice(clusters, n_clusters, replace=False)
        return self.data[self.data[cluster_by].isin(selected_clusters)].reset_index(drop=True)
    
    def convenience_sample(self, n: int) -> pd.DataFrame:
        return self.data.head(n)
    
    def quota_sample(self, stratify_by: str, quotas: Dict[str, int]) -> pd.DataFrame:
        if stratify_by not in self.data.columns:
            raise ValueError(f"Column {stratify_by} not found in data")
        samples = []
        for stratum, quota in quotas.items():
            stratum_data = self.data[self.data[stratify_by] == stratum]
            samples.append(stratum_data.sample(min(quota, len(stratum_data)), random_state=42))
        return pd.concat(samples).reset_index(drop=True)
    
    def snowball_sample(self, seed_size: int, max_samples: int, link_column: str) -> pd.DataFrame:
        if link_column not in self.data.columns:
            raise ValueError(f"Column {link_column} not found in data")
        sampled = self.data.sample(seed_size, random_state=42)
        to_sample = sampled[link_column].tolist()
        while len(sampled) < max_samples and to_sample:
            next_id = to_sample.pop(0)
            new_samples = self.data[self.data[link_column] == next_id]
            sampled = pd.concat([sampled, new_samples]).drop_duplicates().reset_index(drop=True)
            to_sample.extend(new_samples[link_column].tolist())
        return sampled.head(max_samples).reset_index(drop=True)
    
    def multi_stage_sample(self, cluster_by: str, stratify_by: str, n_clusters: int, n_per_stratum: int) -> pd.DataFrame:
        if cluster_by not in self.data.columns or stratify_by not in self.data.columns:
            raise ValueError("Cluster or stratify column not found in data")
        clusters = self.data[cluster_by].unique()
        selected_clusters = np.random.choice(clusters, n_clusters, replace=False)
        cluster_data = self.data[self.data[cluster_by].isin(selected_clusters)]
        return cluster_data.groupby(stratify_by).apply(lambda x: x.sample(
            min(n_per_stratum, len(x)), random_state=42)).reset_index(drop=True)
    
    def oversample(self, stratify_by: str, target_size: int) -> pd.DataFrame:
        if stratify_by not in self.data.columns:
            raise ValueError(f"Column {stratify_by} not found in data")
        current_size = len(self.data)
        if target_size <= current_size:
            return self.data.sample(target_size, random_state=42).reset_index(drop=True)
        else:
            additional_samples = self.data.sample(target_size - current_size, replace=True, random_state=42)
            return pd.concat([self.data, additional_samples]).reset_index(drop=True)
    
    def undersample(self, stratify_by: str, target_size: int) -> pd.DataFrame:
        if stratify_by not in self.data.columns:
            raise ValueError(f"Column {stratify_by} not found in data")
        current_size = len(self.data)
        if target_size >= current_size:
            return self.data.reset_index(drop=True)
        else:
            return self.data.sample(target_size, random_state=42).reset_index(drop=True)


###################################################################################################
###################################################################################################
###################################################################################################
