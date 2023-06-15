import pickle

def main():
    # open a file, where you stored the pickled data
    pomdp_data_file = open('/home/chulabhaya/phd/research/datasets/heavenhell_2/1_million_timesteps/pomdp/6-2-23_hh2_sac_seed_103_time_1683219637_ntw41lb1_global_step_1000000_0_percent_random_data_size_100000_pomdp.pkl', 'rb')
    mdp_data_file = open('/home/chulabhaya/phd/research/datasets/heavenhell_2/1_million_timesteps/mdp/6-2-23_mdp_hh2_no_rec_seed_104_time_1684769262_x2dg6xiv_global_step_1000000_0_percent_random_data_size_200000_mdp.pkl', 'rb')

    # dump information to that file
    pomdp_data = pickle.load(pomdp_data_file)
    mdp_data = pickle.load(mdp_data_file)

    # combine data files
    combined_data = {}
    combined_data["obs"] = pomdp_data["obs"] + mdp_data["obs"]
    combined_data["actions"] = pomdp_data["actions"] + mdp_data["actions"]
    combined_data["next_obs"] = pomdp_data["next_obs"] + mdp_data["next_obs"]
    combined_data["rewards"] = pomdp_data["rewards"] + mdp_data["rewards"]
    combined_data["terminateds"] = pomdp_data["terminateds"] + mdp_data["terminateds"]
    combined_data["truncateds"] = pomdp_data["truncateds"] + mdp_data["truncateds"]
    combined_data["timesteps_in_buffer"] = pomdp_data["timesteps_in_buffer"] + mdp_data["timesteps_in_buffer"]

    # close the file
    pomdp_data_file.close()
    mdp_data_file.close()

    output_data = open(
        "0_percent_random_mdp_0_percent_random_pomdp_combined_size_300000.pkl", "wb"
    )
    pickle.dump(combined_data, output_data)
    output_data.close()


if __name__ == "__main__":
    main()