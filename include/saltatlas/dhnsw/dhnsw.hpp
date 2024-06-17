// Copyright 2020-2024 Lawrence Livermore National Security, LLC and other
// saltatlas Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ygm/comm.hpp>

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
};
}  // namespace saltatlas
