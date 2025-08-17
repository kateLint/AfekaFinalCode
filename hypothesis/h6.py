import pandas as pd
import spacy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model with better error handling
def load_spacy_model(model_name="en_core_web_sm"):
    """Load spaCy model with automatic download if not available."""
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model: {model_name}")
        return nlp
    except OSError:
        logger.warning(f"Model {model_name} not found. Downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
        logger.info(f"Successfully downloaded and loaded: {model_name}")
        return nlp

nlp = load_spacy_model()

# Configuration
CONFIG = {
    'file_path': "/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json",
    'base_output_path': "/Users/kerenlint/Projects/Afeka/all_models/",
    'columns_to_analyze': ["story", "risks-and-challenges"],
    'save_visualizations': True,
    'save_examples': True,
    'num_examples': 5
}

def is_passive(sentence) -> Tuple[bool, List[str]]:
    """
    Detect passive voice in a sentence and return passive constructions.
    
    Returns:
        Tuple[bool, List[str]]: (is_passive, list of passive constructions)
    """
    doc = nlp(sentence)
    passive_constructions = []
    
    for token in doc:
        # Check for passive subject
        if token.dep_ == "nsubjpass":
            passive_constructions.append(f"{token.text} (passive subject)")
        
        # Check for passive auxiliary + past participle
        if token.dep_ == "auxpass" and token.head.tag_ == "VBN":
            passive_constructions.append(f"{token.text} {token.head.text} (aux + past participle)")
        
        # Additional pattern: "by" agent in passive constructions
        if token.text.lower() == "by" and token.dep_ == "agent":
            passive_constructions.append(f"by {' '.join([t.text for t in token.subtree])} (by-agent)")
    
    return len(passive_constructions) > 0, passive_constructions

def analyze_passive(text: str) -> Dict:
    """
    Analyze text for passive voice with detailed statistics.
    
    Returns:
        Dict containing passive analysis results
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'passive_count': 0,
            'total_sentences': 0,
            'passive_ratio': 0.0,
            'passive_examples': [],
            'avg_sentence_length': 0.0
        }
    
    doc = nlp(text)
    results = {
        'passive_count': 0,
        'total_sentences': 0,
        'passive_examples': [],
        'sentence_lengths': [],
        'passive_constructions': []
    }
    
    for sent in doc.sents:
        results['total_sentences'] += 1
        results['sentence_lengths'].append(len(sent.text.split()))
        
        is_passive_sent, constructions = is_passive(sent.text)
        if is_passive_sent:
            results['passive_count'] += 1
            # Store up to 3 examples
            if len(results['passive_examples']) < 3:
                results['passive_examples'].append({
                    'sentence': sent.text.strip(),
                    'constructions': constructions
                })
            results['passive_constructions'].extend(constructions)
    
    # Calculate statistics
    results['passive_ratio'] = results['passive_count'] / max(results['total_sentences'], 1)
    results['avg_sentence_length'] = sum(results['sentence_lengths']) / max(len(results['sentence_lengths']), 1)
    
    return results

def apply_passive_analysis(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Apply passive voice analysis to a DataFrame column with enhanced metrics."""
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in dataset")
        return df.copy()
    
    result_df = df.copy()
    analysis_results = []
    
    logger.info(f"Analyzing '{column_name}' for passive voice...")
    
    for text in tqdm(df[column_name], desc=f"Analyzing {column_name}"):
        analysis_results.append(analyze_passive(text))
    
    # Extract metrics into columns
    result_df[f"{column_name}_passive_sentences"] = [r['passive_count'] for r in analysis_results]
    result_df[f"{column_name}_total_sentences"] = [r['total_sentences'] for r in analysis_results]
    result_df[f"{column_name}_passive_ratio"] = [r['passive_ratio'] for r in analysis_results]
    result_df[f"{column_name}_avg_sentence_length"] = [r['avg_sentence_length'] for r in analysis_results]
    result_df[f"{column_name}_passive_examples"] = [r['passive_examples'] for r in analysis_results]
    
    return result_df

def create_visualizations(df: pd.DataFrame, output_path: str):
    """Create and save visualizations for passive voice analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Passive Voice Analysis Results', fontsize=16)
    
    # 1. Distribution of passive ratios
    for i, col in enumerate(['story', 'risks-and-challenges']):
        if f"{col}_passive_ratio" in df.columns:
            ax = axes[0, i]
            df[f"{col}_passive_ratio"].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of Passive Ratio - {col.title()}')
            ax.set_xlabel('Passive Voice Ratio')
            ax.set_ylabel('Frequency')
            ax.axvline(df[f"{col}_passive_ratio"].mean(), color='red', linestyle='--', label=f'Mean: {df[f"{col}_passive_ratio"].mean():.3f}')
            ax.legend()
    
    # 2. Comparison boxplot
    ax = axes[1, 0]
    passive_data = []
    labels = []
    for col in ['story', 'risks-and-challenges']:
        if f"{col}_passive_ratio" in df.columns:
            passive_data.append(df[f"{col}_passive_ratio"])
            labels.append(col.title())
    
    if passive_data:
        ax.boxplot(passive_data, labels=labels)
        ax.set_title('Passive Voice Ratio Comparison')
        ax.set_ylabel('Passive Voice Ratio')
        ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Sentence length vs Passive ratio
    ax = axes[1, 1]
    if "story_avg_sentence_length" in df.columns and "story_passive_ratio" in df.columns:
        ax.scatter(df["story_avg_sentence_length"], df["story_passive_ratio"], alpha=0.5, label='Story')
    if "risks-and-challenges_avg_sentence_length" in df.columns and "risks-and-challenges_passive_ratio" in df.columns:
        ax.scatter(df["risks-and-challenges_avg_sentence_length"], df["risks-and-challenges_passive_ratio"], 
                  alpha=0.5, label='Risks & Challenges', color='orange')
    ax.set_xlabel('Average Sentence Length (words)')
    ax.set_ylabel('Passive Voice Ratio')
    ax.set_title('Sentence Length vs Passive Voice Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_path, 'passive_voice_visualizations.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualizations saved to: {viz_path}")
    plt.close()

def save_passive_examples(df: pd.DataFrame, output_path: str, num_examples: int = 5):
    """Save examples of passive voice usage to a text file."""
    examples_path = os.path.join(output_path, 'passive_voice_examples.txt')
    
    with open(examples_path, 'w', encoding='utf-8') as f:
        f.write("PASSIVE VOICE EXAMPLES\n")
        f.write("=" * 80 + "\n\n")
        
        for col in ['story', 'risks-and-challenges']:
            if f"{col}_passive_examples" in df.columns:
                f.write(f"\n{col.upper()} EXAMPLES:\n")
                f.write("-" * 40 + "\n\n")
                
                # Get projects with highest passive ratios
                high_passive = df.nlargest(num_examples, f"{col}_passive_ratio")
                
                for idx, row in high_passive.iterrows():
                    f.write(f"Project ID: {idx}\n")
                    f.write(f"Passive Ratio: {row[f'{col}_passive_ratio']:.3f}\n")
                    f.write(f"Examples:\n")
                    
                    for i, example in enumerate(row[f"{col}_passive_examples"][:3], 1):
                        f.write(f"  {i}. {example['sentence']}\n")
                        f.write(f"     Constructions: {', '.join(example['constructions'])}\n\n")
                    
                    f.write("-" * 40 + "\n\n")
    
    logger.info(f"Passive voice examples saved to: {examples_path}")

def generate_summary_report(df: pd.DataFrame, output_path: str):
    """Generate a comprehensive summary report."""
    report_path = os.path.join(output_path, 'passive_voice_summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PASSIVE VOICE ANALYSIS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total projects analyzed: {len(df)}\n\n")
        
        for col in ['story', 'risks-and-challenges']:
            if f"{col}_passive_ratio" in df.columns:
                f.write(f"\n{col.upper()} ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                
                stats = df[f"{col}_passive_ratio"].describe()
                f.write(f"Mean passive ratio: {stats['mean']:.3f}\n")
                f.write(f"Standard deviation: {stats['std']:.3f}\n")
                f.write(f"Minimum: {stats['min']:.3f}\n")
                f.write(f"25th percentile: {stats['25%']:.3f}\n")
                f.write(f"Median: {stats['50%']:.3f}\n")
                f.write(f"75th percentile: {stats['75%']:.3f}\n")
                f.write(f"Maximum: {stats['max']:.3f}\n\n")
                
                # Additional insights
                high_passive = (df[f"{col}_passive_ratio"] > 0.5).sum()
                f.write(f"Projects with >50% passive voice: {high_passive} ({high_passive/len(df)*100:.1f}%)\n")
                
                no_passive = (df[f"{col}_passive_ratio"] == 0).sum()
                f.write(f"Projects with no passive voice: {no_passive} ({no_passive/len(df)*100:.1f}%)\n")
                
                # Correlation with sentence length
                if f"{col}_avg_sentence_length" in df.columns:
                    corr = df[f"{col}_passive_ratio"].corr(df[f"{col}_avg_sentence_length"])
                    f.write(f"Correlation with avg sentence length: {corr:.3f}\n")
    
    logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main execution function."""
    # Check if file exists
    if not os.path.exists(CONFIG['file_path']):
        raise FileNotFoundError(f"File not found at: {CONFIG['file_path']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(CONFIG['base_output_path'], exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from: {CONFIG['file_path']}")
    df = pd.read_json(CONFIG['file_path'], convert_dates=False)
    logger.info(f"Loaded {len(df)} projects")
    
    # Apply analysis to all columns
    combined_df = df.copy()
    for column in CONFIG['columns_to_analyze']:
        combined_df = apply_passive_analysis(combined_df, column)
    
    # Save results
    output_path = os.path.join(CONFIG['base_output_path'], 'passive_voice_analysis.csv')
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Analysis results saved to: {output_path}")
    
    # Generate visualizations
    if CONFIG['save_visualizations']:
        create_visualizations(combined_df, CONFIG['base_output_path'])
    
    # Save examples
    if CONFIG['save_examples']:
        save_passive_examples(combined_df, CONFIG['base_output_path'], CONFIG['num_examples'])
    
    # Generate summary report
    generate_summary_report(combined_df, CONFIG['base_output_path'])
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()