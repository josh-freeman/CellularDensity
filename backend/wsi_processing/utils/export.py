import pandas as pd
import csv
import json
import os
from typing import List, Dict, Any, Optional


def export_results_to_csv(results: List[Dict[str, Any]], 
                         output_path: str,
                         include_metadata: bool = True) -> pd.DataFrame:
    """Export processing results to CSV format."""
    
    # Flatten results for CSV
    rows = []
    for result in results:
        if 'error' in result:
            row = {
                'slide_path': result['slide_path'],
                'status': 'error',
                'error': result['error']
            }
        else:
            row = {
                'slide_path': result['slide_path'],
                'status': 'success',
                'total_nuclei_count': result.get('total_nuclei_count', 0),
                'total_area_mm2': result.get('total_non_background_area_mm2', 0),
                'average_density_per_mm2': result.get('average_density_per_mm2', 0),
                'masks_applied': result.get('masks_applied', 0),
                'contours_processed': result.get('contours_processed', 0),
                'file_name': os.path.basename(result['slide_path']),
                'error': ''
            }
            
            # Add metadata if available
            if include_metadata:
                row['tile_size'] = result.get('parameters', {}).get('tile_size', 0)
                row['contour_level'] = result.get('parameters', {}).get('contour_level', 0)
                row['min_coverage'] = result.get('parameters', {}).get('min_tile_coverage', 0)
            
            # Extract tissue type and grade from filename if possible
            filename = os.path.basename(result['slide_path']).lower()
            row['tissue_type'] = extract_tissue_type(filename)
            row['banff_score'] = extract_banff_score(filename)
            
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def extract_tissue_type(filename: str) -> Optional[str]:
    """Extract tissue type from filename."""
    if 'mucosa' in filename:
        return 'mucosa'
    elif 'skin' in filename:
        return 'skin'
    return None


def extract_banff_score(filename: str) -> Optional[str]:
    """Extract Banff score from filename."""
    if '0-1' in filename:
        return '0.5'
    elif '2-3' in filename:
        return '2.5'
    elif '4-5' in filename:
        return '4.5'
    elif 'banff 1' in filename or 'grade 1' in filename:
        return '1'
    elif 'banff 2' in filename or 'grade 2' in filename:
        return '2'
    elif 'banff 3' in filename or 'grade 3' in filename:
        return '3'
    elif 'banff 4' in filename or 'grade 4' in filename:
        return '4'
    elif 'banff 5' in filename or 'grade 5' in filename:
        return '5'
    elif 'banff 0' in filename or 'grade 0' in filename:
        return '0'
    return None


def export_detailed_results(results: List[Dict[str, Any]], 
                          output_path: str) -> pd.DataFrame:
    """Export detailed results with per-contour information."""
    
    rows = []
    for result in results:
        if 'error' in result:
            continue
            
        detailed_results = result.get('detailed_results', [])
        for i, detail in enumerate(detailed_results):
            row = {
                'slide_path': result['slide_path'],
                'file_name': os.path.basename(result['slide_path']),
                'mask_name': detail.get('mask_name', f'mask_{i}'),
                'contour_index': detail.get('contour_index', i),
                'nuclei_count': detail.get('total_nuclei_count', 0),
                'area_mm2': detail.get('total_area_mm2', 0),
                'density_per_mm2': detail.get('nuclei_density_per_mm2', 0),
                'tiles_processed': detail.get('tiles_processed', 0),
                'tiles_with_tissue': detail.get('tiles_with_tissue', 0),
                'tissue_type': extract_tissue_type(os.path.basename(result['slide_path']).lower()),
                'banff_score': extract_banff_score(os.path.basename(result['slide_path']).lower())
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def create_analysis_report(results: List[Dict[str, Any]], 
                          output_path: str) -> str:
    """Create a text-based analysis report."""
    
    report_lines = []
    report_lines.append("WSI Processing Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Summary statistics
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    report_lines.append(f"Total files processed: {len(results)}")
    report_lines.append(f"Successful: {len(successful_results)}")
    report_lines.append(f"Failed: {len(failed_results)}")
    report_lines.append("")
    
    if successful_results:
        # Calculate aggregate statistics
        total_nuclei = sum(r.get('total_nuclei_count', 0) for r in successful_results)
        total_area = sum(r.get('total_non_background_area_mm2', 0) for r in successful_results)
        avg_density = total_nuclei / total_area if total_area > 0 else 0
        
        report_lines.append("Overall Statistics:")
        report_lines.append(f"  Total nuclei counted: {total_nuclei:,}")
        report_lines.append(f"  Total tissue area: {total_area:.2f} mm²")
        report_lines.append(f"  Average density: {avg_density:.2f} nuclei/mm²")
        report_lines.append("")
        
        # Per-file results
        report_lines.append("Per-file Results:")
        report_lines.append("-" * 30)
        
        for result in successful_results:
            filename = os.path.basename(result['slide_path'])
            nuclei_count = result.get('total_nuclei_count', 0)
            area = result.get('total_non_background_area_mm2', 0)
            density = result.get('average_density_per_mm2', 0)
            
            report_lines.append(f"File: {filename}")
            report_lines.append(f"  Nuclei: {nuclei_count:,}")
            report_lines.append(f"  Area: {area:.2f} mm²")
            report_lines.append(f"  Density: {density:.2f} nuclei/mm²")
            report_lines.append("")
    
    if failed_results:
        report_lines.append("Failed Files:")
        report_lines.append("-" * 20)
        for result in failed_results:
            filename = os.path.basename(result['slide_path'])
            error = result.get('error', 'Unknown error')
            report_lines.append(f"  {filename}: {error}")
        report_lines.append("")
    
    # Write report
    report_content = "\n".join(report_lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    return report_content


def export_configuration(config, output_path: str):
    """Export configuration to JSON file."""
    from ..config.settings import ConfigManager
    
    config_dict = ConfigManager._config_to_dict(config)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def export_all_formats(results: List[Dict[str, Any]], 
                      output_dir: str,
                      config=None) -> Dict[str, str]:
    """Export results in all available formats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    # CSV exports
    summary_csv = os.path.join(output_dir, "summary.csv")
    exported_files['summary_csv'] = summary_csv
    export_results_to_csv(results, summary_csv)
    
    detailed_csv = os.path.join(output_dir, "detailed.csv")
    exported_files['detailed_csv'] = detailed_csv
    export_detailed_results(results, detailed_csv)
    
    # JSON export
    json_path = os.path.join(output_dir, "results.json")
    exported_files['json'] = json_path
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Text report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    exported_files['report'] = report_path
    create_analysis_report(results, report_path)
    
    # Configuration export
    if config:
        config_path = os.path.join(output_dir, "configuration.json")
        exported_files['config'] = config_path
        export_configuration(config, config_path)
    
    return exported_files