// Copyright 2020-2022 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <unordered_map>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/utility.hpp>

namespace saltatlas {
template <typename DistType, typename IndexType, typename Point>
class metric_hyperplane_partitioner {
 public:
  using distance_type     = DistType;
  using id_type           = IndexType;
  using point_type        = Point;
  using tree_node_id_type = uint32_t;

  // Want to avoid implicit conversions between these different id types
  enum class global_tree_node_id_type : tree_node_id_type {};
  enum class intra_level_tree_node_id_type : tree_node_id_type {};
  enum class tree_level_id_type : uint16_t {};
  tree_node_id_type as_int(const global_tree_node_id_type id) {
    return static_cast<tree_node_id_type>(id);
  }
  tree_node_id_type as_int(const intra_level_tree_node_id_type id) {
    return static_cast<tree_node_id_type>(id);
  }
  uint16_t as_int(const tree_level_id_type id) {
    return static_cast<uint16_t>(id);
  }

 private:
  struct tree_node_type {
    id_type       representative;
    distance_type theta_split = std::numeric_limits<distance_type>::max();
  };

  metric_hyperplane_partitioner(const uint32_t num_partitions,
                                hnswlib::SpaceInterface<distance_type> &space,
                                ygm::comm                              &c)
      : m_num_partitions(num_partitions), m_space(space), m_comm(c) {
    // Take 1+log2 to find height of binary tree and fix for incomplete levels
    m_num_levels = ((uint32_t)ceil(log2(m_num_partitions))) + 1;

    m_num_tree_nodes = 2 * m_num_partitions - 1;
    m_tree.resize(m_num_tree_nodes);
  }

  ~metric_hyperplane_partitioner() { m_comm.barrier(); }

  template <class Container>
  void initialize(Container &data) {
    std::unordered_map<id_type, global_tree_node_id_type>
        local_point_assignments;
    std::vector<
        std::vector<id_type>>  // intra_level_tree_id -> vector of points
        local_tree_points;     // For sampling new points for children nodes
    std::vector<id_type>
        candidate_tree_level;  // intra_level_tree_id -> candidate tree nodes
    std::vector<std::vector<distance_type>>
        curr_level_candidate_theta_split;  // intra_level_tree_id -> thetas

    // Assign all items to the root node
    data.for_all(
        [&local_point_assignments, &local_tree_points](const auto &id_point) {
          const auto &[id, point]     = id_point;
          local_point_assignments[id] = 0;
          local_tree_points[0].push_back(id);
        });

    global_tree_node_id_type first_leaf_node =
        m_num_tree_nodes - m_num_partitions + 1;

    for (uint32_t level = 0; level < m_num_levels - 1; ++level) {
      // Find number of tree nodes in current level. Last level may be
      // incomplete.
      intra_level_tree_node_id_type max_nodes_curr_level =
          ((intra_level_tree_node_id_type)1) << level;
      intra_level_tree_node_id_type num_nodes_curr_level =
          std::min<intra_level_tree_node_id_type>(
              max_nodes_curr_level,
              m_num_tree_nodes -
                  ((((intra_level_tree_node_id_type)1) << level) - 1));
      intra_level_tree_node_id_type node_offset_curr_level =
          (((intra_level_tree_node_id_type)1) << level) - 1;

      // Set up the candidate tree level
      candidate_tree_level.clear();
      candidate_tree_level.resize(num_nodes_curr_level);
      candidate_tree_level.clear();
      candidate_tree_level.resize(num_nodes_curr_level);

      for (uint32_t trial = 0; trial < m_candidate_level_trials; ++trial) {
        propose_split_points(candidate_tree_level, local_tree_points);
        compute_split_thetas(local_point_assignments, node_offset_curr_level,
                             num_nodes_curr_level);
      }
      assign_points_to_nodes(local_point_assignments);
    }
  }

 private:
  global_tree_node_id_type level_to_global(tree_level_id_type            level,
                                           intra_level_tree_node_id_type id) {
    return (((intra_level_tree_node_id_type)1) << level) + (id - 1);
  }

  std::pair<tree_level_id_type, intra_level_tree_node_id_type> global_to_level(
      const tree_node_id_type global) {
    tree_level_id_type level = ((tree_level_id_type)floor(log2(global + 1)));
    tree_node_id_type  intra_level_id =
        global - (((global_tree_node_id_type)1) << (level)) + 1;

    return std::make_pair(level, intra_level_id);
  }

  std::pair<global_tree_node_id_type, global_tree_node_id_type> get_child_ids(
      global_tree_node_id_type parent) {
    return std::make_pair(2 * parent + 1, 2 * parent + 2);
  }

  global_tree_node_id_type get_parent_id(global_tree_node_id_type child) {
    return (child - 1) / 2;
  }

  void propose_split_points(
      std::vector<tree_level_id_type>   &candidate_tree_level,
      std::vector<std::vector<id_type>> &local_tree_points) {}

  void compute_split_thetas(
      const std::unordered_map<id_type, global_tree_node_id_type>
                              &point_assignments,
      global_tree_node_id_type node_offset_curr_level,
      global_tree_node_id_type num_nodes_curr_level) {
    for (const auto &[id, tree_node] : point_assignments) {
    }
  }

  void assign_points_to_nodes(
      std::unordered_map<id_type, global_tree_node_id_type>
          &point_assignments) {}

  ygm::comm &m_comm;

  hnswlib::SpaceInterface<distance_type> &m_space;

  std::vector<tree_node_type> m_tree;

  uint32_t                 m_num_partitions;
  global_tree_node_id_type m_num_tree_nodes;
  tree_level_id_type       m_num_levels;
  uint16_t                 m_candidate_level_trials = 10;
};
}  // namespace saltatlas
