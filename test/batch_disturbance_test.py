import os
import json
from datetime import datetime
import hydra
from omegaconf import OmegaConf, DictConfig
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sources.skill.code.eval import (
    eval, 
    filter_recovered_episodes, 
    plot_recovered_cases_only,
    print_recovery_stats
)


def batch_disturbance_test():
    agents = [0, 1, 2]
    magnitudes = [0.1, 0.2, 0.5]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/batch_disturbance/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    for agent_id in agents:
        for magnitude in magnitudes:
            print(f"\n{'='*60}")
            print(f"Testing: Agent {agent_id}, Magnitude {magnitude}")
            print(f"{'='*60}\n")
            
            try:
                import wandb
                if agent_id == agents[0] and magnitude == magnitudes[0]:
                    wandb.init(mode="disabled", project="batch_test")
                
                with hydra.initialize(config_path="sources/skill/1.config/task/eval", version_base=None):
                    cfg = hydra.compose(config_name="default")
                    
                    OmegaConf.set_struct(cfg, False)
                    OmegaConf.set_struct(cfg.environment.env_tweak, False)
                    OmegaConf.set_struct(cfg.model, False)
                    
                    cfg.environment.env_tweak.disturbance_mode = 'adaptive'
                    cfg.environment.env_tweak.disturb_target_agent = agent_id
                    cfg.environment.env_tweak.disturb_magnitude = magnitude
                    cfg.environment.env_tweak.disturb_start_step = 100
                    
                    cfg.environment.env_tweak.tweak_types = []
                    
                    cfg.model.save_group = 'handpicked/mw'
                    
                    cfg.eval_settings.functions.render = False
                    
                    result = eval(cfg)
                    
                    if result:
                        result['agent_id'] = agent_id
                        result['magnitude'] = magnitude
                        all_results.append(result)
                        
                        json_path = os.path.join(
                            results_dir, 
                            f"agent_{agent_id}_mag_{magnitude}.json"
                        )
                        with open(json_path, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        if 'angle_data_grouped' in result:
                            agent_mag_dir = os.path.join(
                                results_dir,
                                f"agent_{agent_id}_mag_{magnitude}"
                            )
                            os.makedirs(agent_mag_dir, exist_ok=True)
                            
                            def save_all_episode_angles(angle_data_grouped, terminate_arr, save_path, disturbance_config):
                                all_episodes = []
                                
                                for idx, (angles, term_step) in enumerate(zip(angle_data_grouped, terminate_arr)):
                                    episode_data = {
                                        'episode_idx': idx,
                                        'term_step': term_step,
                                        'num_angles': len(angles),
                                        'angles': [float(a) for a in angles] if len(angles) > 0 else []
                                    }
                                    all_episodes.append(episode_data)
                                
                                output_data = {
                                    'disturbance_config': disturbance_config,
                                    'total_episodes': len(all_episodes),
                                    'episodes': all_episodes
                                }
                                
                                with open(save_path, 'w') as f:
                                    json.dump(output_data, f, indent=2)
                                
                                with_data = sum(1 for ep in all_episodes if len(ep['angles']) > 0)
                                print(f"✓ Saved all episodes angle data: {save_path}")
                                print(f"  Episodes with data: {with_data}/{len(all_episodes)}")
                            
                            angles_save_path = os.path.join(agent_mag_dir, f"all_angles_agent{agent_id}_mag{magnitude}.json")
                            save_all_episode_angles(
                                result['angle_data_grouped'],
                                result['terminate_arr'],
                                angles_save_path,
                                {
                                    'target_agent': agent_id,
                                    'magnitude': magnitude,
                                    'start_step': 100,
                                    'mode': 'adaptive'
                                }
                            )
                            
                            recovered, episode_details = filter_recovered_episodes(
                                result['angle_data_grouped'],
                                result['terminate_arr'],
                                max_cycles=2000,
                                package_contact_arr=result.get('package_contact_arr', [])
                            )
                            
                            filter_log_filename = f"filter_log_agent{agent_id}_mag{magnitude}.json"
                            filter_log_path = os.path.join(agent_mag_dir, filter_log_filename)
                            
                            filter_log_data = {
                                'disturbance_config': {
                                    'target_agent': agent_id,
                                    'magnitude': magnitude,
                                    'start_step': 100,
                                    'mode': 'adaptive',
                                    'failure_threshold_deg': 6.5,
                                    'recovery_threshold_deg': 5.0
                                },
                                'summary': {
                                    'total_episodes': len(episode_details),
                                    'recovered_count': len(recovered),
                                    'filtered_count': len([e for e in episode_details if e['status'] == 'filtered'])
                                },
                                'episode_details': episode_details
                            }
                            
                            with open(filter_log_path, 'w') as f:
                                json.dump(filter_log_data, f, indent=2)
                            
                            angles_data_filename = f"full_angles_agent{agent_id}_mag{magnitude}.json"
                            angles_data_path = os.path.join(agent_mag_dir, angles_data_filename)
                            
                            full_angle_episodes = []
                            for ep in episode_details:
                                if 'angles' in ep:
                                    full_angle_episodes.append({
                                        'episode_idx': ep['episode_idx'],
                                        'status': ep['status'],
                                        'filter_reason': ep['filter_reason'],
                                        'angles': ep['angles'],
                                        'metadata': {
                                            'max_angle': ep['max_angle'],
                                            'first_unstable_step': ep['first_unstable_step'],
                                            'last_unstable_step': ep['last_unstable_step'],
                                            'recovery_step': ep.get('recovery_step')
                                        }
                                    })
                            
                            if full_angle_episodes:
                                with open(angles_data_path, 'w') as f:
                                    json.dump({
                                        'disturbance_config': filter_log_data['disturbance_config'],
                                        'total_episodes_with_angles': len(full_angle_episodes),
                                        'episodes': full_angle_episodes
                                    }, f, indent=2)
                                print(f"✓ Full angle data saved: {angles_data_path} ({len(full_angle_episodes)} episodes)")
                            
                            plot_recovered_cases_only(recovered, agent_mag_dir, {
                                'target_agent': agent_id,
                                'magnitude': magnitude
                            })
                            
                            print_recovery_stats(recovered)
                        
                        print(f"✓ Test completed for Agent {agent_id}, Magnitude {magnitude}")
                    
                hydra.core.global_hydra.GlobalHydra.instance().clear()
                
            except Exception as e:
                print(f"✗ Error testing Agent {agent_id}, Magnitude {magnitude}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    generate_summary_report(all_results, results_dir)
    
    print(f"\n{'='*60}")
    print(f"Batch testing completed!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}\n")
    
    return results_dir


def generate_summary_report(results, output_dir):
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISTURBANCE TEST SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Tests: {len(results)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for r in results:
            f.write(f"Agent {r['agent_id']}, Magnitude {r['magnitude']}:\n")
            f.write(f"  MTTF: {r.get('disturbance_mttf_avg', 'N/A')}\n")
            f.write(f"  Recovery Time: {r.get('disturbance_recovery_time_avg', 'N/A')}\n")
            
            max_angle = r.get('disturbance_max_angle', None)
            if max_angle is not None:
                f.write(f"  Max Angle: {max_angle:.2f}°\n")
            else:
                f.write(f"  Max Angle: N/A\n")
            
            f.write(f"  Terminate Count: {r.get('terminate_cnt', 'N/A')}\n")
            f.write(f"  Avg Terminate At: {r.get('avg_terminate_at', 'N/A')}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for agent_id in [0, 1, 2]:
            agent_results = [r for r in results if r['agent_id'] == agent_id]
            if agent_results:
                f.write(f"Agent {agent_id}:\n")
                mttf_values = [r.get('disturbance_mttf_avg') for r in agent_results if r.get('disturbance_mttf_avg') is not None]
                if mttf_values:
                    f.write(f"  Avg MTTF across magnitudes: {sum(mttf_values)/len(mttf_values):.2f}\n")
                f.write("\n")
    
    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    batch_disturbance_test()
