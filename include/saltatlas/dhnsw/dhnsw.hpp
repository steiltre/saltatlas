// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/comm.hpp>

#include <saltatlas/partitioner/metric_hyperplane_partitioner.hpp>
#include <saltatlas/point_store.hpp>

namespace saltatlas {

template <typename Id, typename Point, typename Distance = double,
          typename Allocator = std::allocator<std::byte>>
class dhnsw {
 public:
  using id_type          = Id;
  using point_type       = Point;
  using distance_type    = Distance;
  using allocator_type   = Allocator;
  using point_store_type = point_store<id_type, point_type, std::hash<id_type>,
                                       std::equal_to<>, allocator_type>;
  using distance_function_type =
      std::function<distance_type(const point_type &, const point_type &)>;

 private:
  using point_partitioner =
      saltatlas::metric_hyperplane_partitioner<distance_type, id_type,
                                               point_type>;

 public:
  dhnsw(const distance_function_type &distance_func, ygm::comm &comm,
        const uint32_t max_voronoi_rank, const uint32_t num_partitions)
      : m_comm(comm) {}

 private:
  ygm::comm &m_comm;
  // dhnsw_detail::dhnsw_impl<distance_type, id_type, point_type,
  // point_partitioner>
  // m_index_impl;
  // dhnsw_detail::query_engine_impl<distance_type, id_type, point_type,
  // point_partitioner>
  // m_query_engine_impl;

  point_store_type m_point_store;
};
}  // namespace saltatlas
