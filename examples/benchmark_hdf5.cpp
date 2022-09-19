// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <string>

#include <ygm/comm.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility.hpp>

#include <saltatlas/container/pair_bag.hpp>
#include <saltatlas/dhnsw/detail/utility.hpp>
#include <saltatlas/dhnsw/dhnsw.hpp>
#include <saltatlas/partitioner/metric_hyperplane_partitioner.hpp>
#include <saltatlas/partitioner/voronoi_partitioner.hpp>

#include <saltatlas_h5_io/h5_reader.hpp>
#include <saltatlas_h5_io/h5_writer.hpp>

void usage(ygm::comm &comm) {
  if (comm.rank0()) {
    std::cerr
        << "Usage: -v <int> -h <int> -k <int> -i <string>\n"
        << " -v <int>   - Voronoi rank (required)\n"
        << " -s <int>   - Number of seeds (required)\n"
        << " -p <int>   - Number of hops\n"
        << " -k <int>      - Number of nearest neighbors\n"
        << " -i <string> 	 - File containing data to build index from "
           "(required)\n"
        << " -q <string>   - File containing data to query index with\n"
        << " -g <string>   - File containing ground truth nearest neighbors\n"
        << " -h            - print help and exit\n\n";
  }
}

void parse_cmd_line(int argc, char **argv, ygm::comm &comm, int &voronoi_rank,
                    int &num_hops, int &num_seeds, int &k,
                    std::string &index_filename, std::string &query_filename,
                    std::string &ground_truth_filename) {
  if (comm.rank0()) {
    std::cout << "CMD line:";
    for (int i = 0; i < argc; ++i) {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;
  }

  bool found_index_filename{false};
  bool found_query_filename{false};
  bool found_ground_truth_filename{false};

  voronoi_rank          = -1;
  num_hops              = -1;
  num_seeds             = -1;
  k                     = -1;
  index_filename        = "";
  query_filename        = "";
  ground_truth_filename = "";

  int  c;
  bool prn_help = false;
  while ((c = getopt(argc, argv, "v:p:s:k:i:q:g:h ")) != -1) {
    switch (c) {
      case 'h':
        prn_help = true;
        break;
      case 'v':
        voronoi_rank = atoi(optarg);
        break;
      case 's':
        num_seeds = atoi(optarg);
        break;
      case 'p':
        num_hops = atoi(optarg);
        break;
      case 'k':
        k = atoi(optarg);
        break;
      case 'i':
        found_index_filename = true;
        index_filename       = optarg;
        break;
      case 'q':
        found_query_filename = true;
        query_filename       = optarg;
        break;
      case 'g':
        found_ground_truth_filename = true;
        ground_truth_filename       = optarg;
        break;
      default:
        std::cerr << "Unrecognized option: " << c << ", ignore." << std::endl;
        prn_help = true;
        break;
    }
  }

  // Detect misconfigured options
  if (!found_index_filename) {
    comm.cout0("Must specify file to build index from");
    prn_help = true;
  }
  if (voronoi_rank < 1) {
    comm.cout0(
        "Must specify positive Voronoi rank to build distributed index with");
    prn_help = true;
  }
  if (num_seeds < 1) {
    comm.cout0(
        "Must specify a positive number of seeds to use for building "
        "distributed index");
    prn_help = true;
  }
  if (found_query_filename && ((num_hops < 0) || (k < 1))) {
    comm.cout0(
        "Must specify positive number of hops and number of nearest "
        "neighbors to query index with");
    prn_help = true;
  }
  if (found_ground_truth_filename && !found_query_filename) {
    comm.cout0("Must specify a file of query points to use ground truth");
    prn_help = true;
  }

  if (prn_help) {
    usage(comm);
    exit(-1);
  }
}

// Generates a vector of column names of the form "colxxx"
std::vector<std::string> generic_column_names(int n) {
  std::vector<std::string> to_return;

  for (int i = 0; i < n; ++i) {
    std::string to_push{"col"};
    if (i < 100) {
      to_push += "0";
    }
    if (i < 10) {
      to_push += "0";
    }
    to_push += std::to_string(i);
    to_return.push_back(to_push);
  }

  return to_return;
}

float my_l2_sqr(const std::vector<float> &x, const std::vector<float> &y) {
  if (x.size() != y.size()) {
    std::cerr << "Size mismatch for l2 distance" << std::endl;
    exit;
  }

  float dist_sqr{0.0};

  for (size_t i = 0; i < x.size(); ++i) {
    dist_sqr += (x[i] - y[i]) * (x[i] - y[i]);
  }

  return dist_sqr;
}

float my_l2(const std::vector<float> &x, const std::vector<float> &y) {
  return sqrt(my_l2_sqr(x, y));
}

template <typename IndexType, typename Point>
ygm::container::pair_bag<IndexType, Point> read_data(
    ygm::container::bag<std::string> &bag_filenames,
    const std::vector<std::string>   &data_col_names) {
  ygm::container::pair_bag<IndexType, Point> to_return(bag_filenames.comm());

  auto read_file_lambda = [&to_return, &data_col_names](const auto &fname) {
    saltatlas::h5_io::h5_reader reader(fname);
    if (!reader.is_open()) {
      std::cerr << "Failed to open " << fname << std::endl;
      exit(EXIT_FAILURE);
    }

    const auto data_indices = reader.read_column<IndexType>("index");
    const auto data = reader.read_columns_row_wise<float>(data_col_names);
    for (int j = 0; j < data.size(); ++j) {
      to_return.async_insert(std::make_pair(data_indices[j], data[j]));
    }
  };

  bag_filenames.for_all(read_file_lambda);

  return to_return;
}

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
void build_index(
    ygm::container::bag<std::string>                          &bag_filenames,
    saltatlas::dhnsw<DistType, IndexType, Point, Partitioner> &dist_index,
    Partitioner<DistType, IndexType, Point>                   &partitioner,
    const size_t num_seeds, const std::vector<std::string> &data_col_names) {
  if (dist_index.comm().rank0()) {
    std::cout << "\n****Building distributed index****"
              << "\nNumber of Voronoi cells: " << num_seeds << std::endl;
  }

  ygm::timer step_timer{};

  dist_index.comm().cout0("Reading data to temporary bag");
  auto bag_data = read_data<IndexType, Point>(bag_filenames, data_col_names);
  dist_index.comm().barrier();
  dist_index.comm().cout0("Data reading time: ", step_timer.elapsed());

  dist_index.partition_data(bag_data, num_seeds);

  dist_index.comm().cout0("Distributing data across ranks");

  auto add_point_lambda = [&dist_index, &bag_data](const auto &index_point) {
    dist_index.queue_data_point_insertion(index_point.first,
                                          index_point.second);
  };

  dist_index.comm().barrier();
  step_timer.reset();
  bag_data.for_all(add_point_lambda);

  dist_index.comm().barrier();
  dist_index.comm().cout0("Data distribution time: ", step_timer.elapsed());
  step_timer.reset();

  dist_index.comm().cout0("Initializing per-cell HNSW structures");
  dist_index.initialize_hnsw();
  dist_index.comm().barrier();
  dist_index.comm().cout0("HNSW initialization time: ", step_timer.elapsed());

  size_t global_hnsw_size = dist_index.global_size();
  dist_index.comm().cout0("Global HNSW size: ", global_hnsw_size);

  dist_index.comm().cout0("Finished creating index structure");
}

template <typename DistType, typename IndexType, typename Point,
          template <typename, typename, typename> class Partitioner>
void benchmark_query_trial(
    int voronoi_rank, int hops, int k,
    saltatlas::dhnsw<DistType, IndexType, Point, Partitioner> &dist_index,
    ygm::container::bag<std::string>                          &bag_query_files,
    const std::vector<std::string>                            &data_col_names) {
  if (dist_index.comm().rank() == 0) {
    std::cout << "\n****Beginning query trial****"
              << "\nVoronoi rank: " << voronoi_rank << "\nHops: " << hops
              << "\nNearest neighbors: " << k << std::endl;
  }

  static int num_queries;
  num_queries = 0;

  auto empty_lambda =
      [](const Point                              &query_pt,
         const std::multimap<DistType, IndexType> &nearest_neighbors,
         auto                                      dhnsw) { ++num_queries; };

  auto perform_query_lambda = [&empty_lambda, &dist_index, &hops, &voronoi_rank,
                               &k](const auto &index_point) {
    dist_index.query(index_point.second, k, hops, voronoi_rank, 1,
                     empty_lambda);
  };

  auto bag_query_points =
      read_data<IndexType, Point>(bag_query_files, data_col_names);

  bag_query_points.for_all(perform_query_lambda);

  dist_index.comm().barrier();

  dist_index.comm().cout0("Total query points: ",
                          dist_index.comm().all_reduce_sum(num_queries));

  return;
}

template <typename DHNSW>
void benchmark_query_trial_ground_truth(
    int voronoi_rank, int hops, int k, DHNSW &dist_index,
    ygm::container::bag<std::string> &bag_query_files,
    ygm::container::bag<std::string> &bag_ground_truth_files,
    const std::vector<std::string>   &data_col_names) {
  using dist_t  = typename DHNSW::dist_t;
  using index_t = typename DHNSW::index_t;
  using point_t = typename DHNSW::point_t;

  ygm::container::map<index_t, std::vector<index_t>> ground_truth(
      dist_index.comm());

  if (dist_index.comm().rank() == 0) {
    std::cout << "\n****Beginning query trial with ground truth****"
              << "\nVoronoi rank: " << voronoi_rank << "\nHops: " << hops
              << "\nNearest neighbors: " << k << std::endl;
  }

  // Read ground truth
  auto read_ground_truth_file_lambda = [&ground_truth, &k,
                                        &dist_index](const auto &fname) {
    saltatlas::h5_io::h5_reader reader(fname);

    if (!reader.is_open()) {
      std::cerr << "Failed to open " << fname << std::endl;
      exit(EXIT_FAILURE);
    }

    // One column for indices
    auto num_ground_truth_nearest_neighbors = reader.column_names().size() - 1;

    if (num_ground_truth_nearest_neighbors < k) {
      dist_index.comm().cout0(
          "Searching for more nearest neighbors than ground truth contains");
      exit(EXIT_FAILURE);
    }

    auto nearest_neighbor_col_names =
        generic_column_names(num_ground_truth_nearest_neighbors);

    const auto indices = reader.read_column<index_t>("index");
    const auto ground_truth_part =
        reader.read_columns_row_wise<index_t>(nearest_neighbor_col_names);

    for (int i = 0; i < ground_truth_part.size(); ++i) {
      std::vector<index_t> truncated_ground_truth;
      for (int j = 0; j < k; ++j) {
        truncated_ground_truth.push_back(ground_truth_part[i][j]);
      }
      ground_truth.async_insert(indices[i], truncated_ground_truth);
    }
  };

  dist_index.comm().cout0("\nReading ground truth into YGM map");
  bag_ground_truth_files.for_all(read_ground_truth_file_lambda);

  static uint64_t true_positives;
  static uint64_t total_neighbors;
  true_positives  = 0;
  total_neighbors = 0;

  // Find approximate nearest neighbors, then check against ground truth values
  // stored in distributed map
  auto query_nearest_neighbors_lambda =
      [](const point_t                        &query_pt,
         const std::multimap<dist_t, index_t> &nearest_neighbors, auto dhnsw,
         uint64_t data_index, auto ground_truth_ptr) {
        // Lambda to check ANN against ground truth
        auto check_nearest_neighbors_lambda = [](auto &index_gt, auto ann) {
          auto intersection_lambda = [](auto &vec1, auto &vec2) {
            int to_return{0};

            std::sort(vec1.begin(), vec1.end());
            std::sort(vec2.begin(), vec2.end());

            auto vec1_iter = vec1.begin();
            auto vec2_iter = vec2.begin();

            while ((vec1_iter != vec1.end()) && (vec2_iter != vec2.end())) {
              if (*vec1_iter == *vec2_iter) {
                ++to_return;
                ++vec1_iter;
                ++vec2_iter;
              } else if (*vec1_iter < *vec2_iter) {
                ++vec1_iter;
              } else {
                ++vec2_iter;
              }
            }
            return to_return;
          };

          true_positives += intersection_lambda(index_gt.second, ann);
          total_neighbors += ann.size();
        };

        // Put approximate nearest neighbors into vector
        std::vector<index_t> ann_vec;

        for (const auto &dist_ngbr : nearest_neighbors) {
          ann_vec.push_back(dist_ngbr.second);
        }

        ground_truth_ptr->async_visit(data_index,
                                      check_nearest_neighbors_lambda, ann_vec);
      };

  auto perform_query_lambda = [&query_nearest_neighbors_lambda, &dist_index,
                               &hops, &voronoi_rank, &k,
                               &ground_truth](const auto &index_point) {
    dist_index.query(index_point.second, k, hops, voronoi_rank, 1,
                     query_nearest_neighbors_lambda, index_point.first,
                     ground_truth.get_ygm_ptr());
  };

  dist_index.comm().cout0("Reading query points into bag");
  auto bag_query_points =
      read_data<index_t, point_t>(bag_query_files, data_col_names);

  bag_query_points.for_all(perform_query_lambda);

  dist_index.comm().barrier();

  float recall = dist_index.comm().all_reduce_sum(true_positives) /
                 ((float)dist_index.comm().all_reduce_sum(total_neighbors));
  dist_index.comm().cout0("Recall: ", recall);

  return;
}

void fill_filenames_bag(ygm::container::bag<std::string> &bag,
                        const std::string                &filename) {
  // Test file to see if it is HDF5
  if (bag.comm().rank0()) {
    saltatlas::h5_io::h5_reader reader(filename);

    if (!reader.is_open()) {
      std::cout << "Could not open " << filename
                << " as HDF5. Assuming it contains a list of HDF5 files"
                << std::endl;

      std::ifstream ifs(filename.c_str());
      std::string   line;
      while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string       fname;
        ss >> fname;
        bag.async_insert(fname);
      }
    } else {
      bag.async_insert(filename);
    }
  }

  bag.comm().barrier();

  return;
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);
  {
    ygm::timer step_timer{};

    int mpi_rank = world.rank();
    int mpi_size = world.size();

    int voronoi_rank;
    int hops;
    int num_seeds;
    int k;

    std::string index_filename;
    std::string query_filename;
    std::string ground_truth_filename;

    parse_cmd_line(argc, argv, world, voronoi_rank, hops, num_seeds, k,
                   index_filename, query_filename, ground_truth_filename);

    ygm::container::bag<std::string> bag_index_filenames(world);

    fill_filenames_bag(bag_index_filenames, index_filename);

    // Build index
    auto my_l2_space = saltatlas::dhnsw_detail::SpaceWrapper(my_l2_sqr);

    saltatlas::metric_hyperplane_partitioner<float, std::size_t,
                                             std::vector<float>>
                     partitioner(world, my_l2_space);
    saltatlas::dhnsw dist_index(voronoi_rank, num_seeds, &my_l2_space, &world,
                                partitioner);

    // extra column for indices
    int num_cols =
        saltatlas::dhnsw_detail::get_num_columns(bag_index_filenames, world) -
        1;
    auto data_col_names = generic_column_names(num_cols);
    step_timer.reset();
    build_index(bag_index_filenames, dist_index, partitioner, num_seeds,
                data_col_names);
    dist_index.comm().barrier();
    world.cout0("Total index construction time: ", step_timer.elapsed());

    // Time querying if query file is provided
    if ((query_filename != "") && (ground_truth_filename == "")) {
      ygm::container::bag<std::string> bag_query_filenames(world);
      fill_filenames_bag(bag_query_filenames, query_filename);

      world.barrier();
      step_timer.reset();
      benchmark_query_trial(voronoi_rank, hops, k, dist_index,
                            bag_query_filenames, data_col_names);

      auto     query_time = step_timer.elapsed();
      uint64_t num_queries =
          saltatlas::dhnsw_detail::count_points_hdf5(bag_query_filenames);

      world.cout0("Query time: ", query_time);
      world.cout0("\t", num_queries / query_time, " queries per second");
    }

    // Time querying and recall if query file and ground truth provided
    if ((query_filename != "") && (ground_truth_filename != "")) {
      ygm::container::bag<std::string> bag_query_filenames(world);
      fill_filenames_bag(bag_query_filenames, query_filename);

      ygm::container::bag<std::string> bag_ground_truth_filenames(world);
      fill_filenames_bag(bag_ground_truth_filenames, ground_truth_filename);

      world.barrier();
      step_timer.reset();
      benchmark_query_trial_ground_truth(
          voronoi_rank, hops, k, dist_index, bag_query_filenames,
          bag_ground_truth_filenames, data_col_names);

      auto     query_time = step_timer.elapsed();
      uint64_t num_queries =
          saltatlas::dhnsw_detail::count_points_hdf5(bag_query_filenames);

      world.cout0("Query time: ", query_time);
      world.cout0("\t", num_queries / query_time, " queries per second");
    }
  }
}
