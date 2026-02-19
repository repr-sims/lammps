/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Paul Crozier (SNL)
                         Alexander Stukowski
------------------------------------------------------------------------- */

#include "fix_atom_swap.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "graphics.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "random_park.h"
#include "region.h"
#include "suffix.h"
#include "update.h"

#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAtomSwap::FixAtomSwap(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), type_list(nullptr), mu(nullptr),
    qtype(nullptr), mtype(nullptr), sqrt_mass_ratio(nullptr), local_swap_iatom_list(nullptr),
    local_swap_jatom_list(nullptr), local_swap_atom_list(nullptr), local_swap_igroup(nullptr),
    local_swap_jgroup(nullptr), is_affected(nullptr), affected_list(nullptr),
    local_energy_cache(nullptr), local_energy_new(nullptr),
    random_equal(nullptr), random_unequal(nullptr), c_pe(nullptr),
    imgobjs(nullptr), imgparms(nullptr), list(nullptr), max_affected(0)
{

  if (narg < 10) utils::missing_cmd_args(FLERR, "fix atom/swap", error);

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE) {
    vizsteps = 0;
  } else {
    vizsteps = 1000;
  }

  // required args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);
  double temperature = utils::numeric(FLERR, arg[6], false, lmp);

  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix atom/swap command nevery value");
  if (ncycles < 0) error->all(FLERR, 4, "Illegal fix atom/swap command ncycles value");
  if (seed <= 0) error->all(FLERR, 5, "Illegal fix atom/swap command random seed");
  if (temperature <= 0.0) error->all(FLERR, 6, "Illegal fix atom/swap command temperature value");

  beta = 1.0 / (force->boltz * temperature);

  memory->create(type_list, atom->ntypes, "atom/swap:type_list");
  memory->create(mu, atom->ntypes + 1, "atom/swap:mu");
  for (int i = 0; i <= atom->ntypes; i++) mu[i] = 0.0;

  // initialize local energy optimization members BEFORE options()
  use_local_energy = false;
  local_energy_mode = LOCAL_AUTO;


  // default group_size is 1 (swap single atoms)
  group_size = 1;

  // read options from end of input line

  options(narg - 7, &arg[7]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp, seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp, seed);

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  mc_active = 0;

  nswap_attempts = 0.0;
  nswap_successes = 0.0;

  atom_swap_nmax = 0;
  local_swap_atom_list = nullptr;
  local_swap_iatom_list = nullptr;
  local_swap_jatom_list = nullptr;
  is_affected = nullptr;
  affected_list = nullptr;
  max_affected = 0;

  // set comm size needed by this Fix

  if (atom->q_flag)
    comm_forward = 2;
  else
    comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixAtomSwap::~FixAtomSwap()
{
  memory->destroy(type_list);
  memory->destroy(mu);
  memory->destroy(qtype);
  memory->destroy(mtype);
  memory->destroy(sqrt_mass_ratio);
  memory->destroy(local_swap_iatom_list);
  memory->destroy(local_swap_jatom_list);
  memory->destroy(local_swap_igroup);
  memory->destroy(local_swap_jgroup);
  memory->destroy(is_affected);
  memory->destroy(affected_list);
  memory->destroy(local_energy_cache);
  memory->destroy(local_energy_new);
  delete[] idregion;
  delete random_equal;
  delete random_unequal;
  memory->destroy(imgobjs);
  memory->destroy(imgparms);
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixAtomSwap::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix atom/swap command");

  ke_flag = 1;
  semi_grand_flag = 0;
  nswaptypes = 0;
  nmutypes = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region) error->all(FLERR, "Region {} for fix atom/swap does not exist", arg[iarg + 1]);
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      ke_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "semi-grand") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      semi_grand_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "types") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        if (nswaptypes >= atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        type_list[nswaptypes] = utils::expand_type_int(FLERR, arg[iarg], Atom::ATOM, lmp);
        nswaptypes++;
        iarg++;
      }
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        nmutypes++;
        if (nmutypes > atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        mu[nmutypes] = utils::numeric(FLERR, arg[iarg], false, lmp);
        iarg++;
      }
    } else if (strcmp(arg[iarg], "group_size") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      group_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (group_size < 1) error->all(FLERR, "Illegal fix atom/swap group_size value");
      iarg += 2;
    } else if (strcmp(arg[iarg], "local") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        local_energy_mode = LOCAL_YES;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        local_energy_mode = LOCAL_NO;
      else if (strcmp(arg[iarg + 1], "auto") == 0)
        local_energy_mode = LOCAL_AUTO;
      else
        error->all(FLERR, "Illegal fix atom/swap local value: must be yes/no/auto");
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix atom/swap command");
  }
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"vizsteps") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify atom/swap", error);
    vizsteps = utils::inumeric(FLERR, arg[1], false, lmp);
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::init()
{
  if ((atom->mass != nullptr) && (atom->rmass != nullptr) && (comm->me == 0))
    error->warning(FLERR, "Fix atom/swap will use per-atom masses for velocity rescaling");

  c_pe = modify->get_compute_by_id("thermo_pe");

  int *type = atom->type;

  if (nswaptypes < 2)
    error->all(FLERR, Error::NOLASTLINE,
               "Must specify at least 2 atom types in fix atom/swap command");

  if (semi_grand_flag) {
    if (nswaptypes != nmutypes)
      error->all(FLERR, Error::NOLASTLINE, "Need nswaptypes mu values in fix atom/swap command");
    if (group_size > 1)
      error->all(FLERR, Error::NOLASTLINE,
                 "group_size > 1 not supported in semi-grand mode");
  } else {
    if (nswaptypes != 2)
      error->all(FLERR, Error::NOLASTLINE,
                 "Exactly 2 atom types must be used without semi-grand keyword in fix atom/swap");
    if (nmutypes != 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "Mu not allowed when not using semi-grand in fix atom/swap command");
  }

  // must have a pair style and not use INTEL package

  if (!force->pair) error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap requires a pair style");
  if (force->pair && (force->pair->suffix_flag & Suffix::INTEL))
    error->all(FLERR, Error::NOLASTLINE, "Fix {} is not compatible with /intel pair styles", style);

  // check if constraints for hybrid pair styles are fulfilled

  if (utils::strmatch(force->pair_style, "^hybrid")) {
    auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
    if (hybrid) {
      for (int i = 0; i < nswaptypes - 1; ++i) {
        int type1 = type_list[i];
        for (int j = i + 1; j < nswaptypes; ++j) {
          int type2 = type_list[j];
          if (hybrid->nmap[type1][type1] != hybrid->nmap[type2][type2])
            error->all(FLERR, Error::NOLASTLINE,
                       "Pair {} substyles for atom types {} and {} are not compatible",
                       force->pair_style, type1, type2);
          for (int k = 0; k < hybrid->nmap[type1][type1]; ++k) {
            if (hybrid->map[type1][type1][k] != hybrid->map[type2][type2][k])
              error->all(FLERR, Error::NOLASTLINE,
                         "Pair {} substyles for atom types {} and {} are not compatible",
                         force->pair_style, type1, type2);
          }
        }
      }
    }
  }

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region)
      error->all(FLERR, Error::NOLASTLINE, "Region {} for fix atom/swap does not exist", idregion);
  }

  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    if (type_list[iswaptype] <= 0 || type_list[iswaptype] > atom->ntypes)
      error->all(FLERR, "Invalid atom type in fix atom/swap command");

  // this is only required for non-semi-grand
  // in which case, nswaptypes = 2

  if (atom->q_flag && !semi_grand_flag) {
    double qmax, qmin;
    int firstall, first;
    memory->create(qtype, nswaptypes, "atom/swap:qtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first) {
              qtype[iswaptype] = atom->q[i];
              first = 0;
            } else if (qtype[iswaptype] != atom->q[i])
              error->one(FLERR, "All atoms of a swapped type must have the same charge.");
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall)
        error->all(FLERR,
                   "At least one atom of each swapped type must be present to define charges.");
      if (first) qtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) qtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (qmax != qmin) error->all(FLERR, "All atoms of a swapped type must have same charge.");
      qtype[iswaptype] = qmax;
    }
  }

  // if we have per-atom masses, check that rmass is consistent with type,
  // and set per-type mass to that value
  if ((atom->rmass !=  nullptr) && !semi_grand_flag) {
    double mmax, mmin;
    int firstall, first;
    memory->create(mtype, nswaptypes, "atom/swap:mtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first > 0) {
              mtype[iswaptype] = atom->rmass[i];
              first = 0;
            } else if (mtype[iswaptype] != atom->rmass[i])
              first = -1;
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall < 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "All atoms of a swapped type must have the same per-atom mass");
      if (firstall > 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "At least one atom of each swapped type must be present to define masses");
      if (first) mtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) mtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (mmax != mmin)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have same mass.");
      mtype[iswaptype] = mmax;
    }
  }

  memory->create(sqrt_mass_ratio, atom->ntypes + 1, atom->ntypes + 1, "atom/swap:sqrt_mass_ratio");
  if (atom->rmass != nullptr) {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) sqrt_mass_ratio[itype][jtype] = 1.0;
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      int itype = type_list[iswaptype];
      for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++) {
        int jtype = type_list[jswaptype];
        sqrt_mass_ratio[itype][jtype] = sqrt(mtype[iswaptype] / mtype[jswaptype]);
      }
    }
  } else {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++)
        sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype] / atom->mass[jtype]);
  }

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  // check that no swappable atoms are in atom->firstgroup
  // swapping such an atom might not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);

    if (flagall) error->all(FLERR, "Cannot do atom/swap on atoms in atom_modify first group");
  }

  // allocate arrays for group swaps
  memory->create(local_swap_igroup, group_size, "atom/swap:local_swap_igroup");
  memory->create(local_swap_jgroup, group_size, "atom/swap:local_swap_jgroup");

  // determine if local energy calculation should be used
  // user can force via 'local yes/no', or use 'local auto' for auto-detection
  // requires: pair style supports atomic energy, no kspace with charge change,
  // no tail corrections, equal cutoffs, and atomic (non-molecular) system

  // Handle user preference: LOCAL_NO, LOCAL_YES, or LOCAL_AUTO
  if (local_energy_mode == LOCAL_NO) {
    use_local_energy = false;
    if (comm->me == 0)
      utils::logmesg(lmp, "Fix atom/swap: using FULL energy calculation (user specified)\n");
    return;  // skip auto-detection
  } else if (local_energy_mode == LOCAL_YES) {
    use_local_energy = true;  // will validate below
  } else {
    use_local_energy = true;  // LOCAL_AUTO: will auto-detect below
  }

  if (!force->pair || !force->pair->atomic_energy_enable) {
    if (local_energy_mode == LOCAL_YES)
      error->all(FLERR, "Fix atom/swap local yes requires pair style with atomic_energy_enable");
    use_local_energy = false;
  }

  if (force->kspace && nswaptypes >= 2) {
    // kspace only matters if charges differ between swapped types
    bool charges_differ = false;
    if (atom->q_flag && qtype) {
      for (int s1 = 0; s1 < nswaptypes && !charges_differ; s1++)
        for (int s2 = s1+1; s2 < nswaptypes && !charges_differ; s2++)
          if (qtype[s1] != qtype[s2]) charges_differ = true;
    }
    if (charges_differ) {
      if (local_energy_mode == LOCAL_YES)
        error->all(FLERR, "Fix atom/swap local yes incompatible with kspace and differing charges");
      use_local_energy = false;
    }
  }

  if (force->pair && force->pair->tail_flag) {
    if (local_energy_mode == LOCAL_YES)
      error->all(FLERR, "Fix atom/swap local yes incompatible with tail corrections");
    use_local_energy = false;
  }

  if (unequal_cutoffs) {
    if (local_energy_mode == LOCAL_YES)
      error->all(FLERR, "Fix atom/swap local yes incompatible with unequal cutoffs");
    use_local_energy = false;
  }

  if (atom->molecular != Atom::ATOMIC) {
    if (local_energy_mode == LOCAL_YES)
      error->all(FLERR, "Fix atom/swap local yes requires atomic system (no bonds/angles)");
    use_local_energy = false;
  }

  if (use_local_energy) {
    if (atom->map_style == 0)
      error->all(FLERR, "Fix atom/swap local energy optimization requires an atom map, use atom_modify map array");

    // Request a full neighbor list for the fix
    // This is required to identify all atoms 'k' that have 'i' as a neighbor on ALL processors.
    neighbor->add_request(this, NeighConst::REQ_FULL);

    if (comm->me == 0) {
      const char *mode_str = (local_energy_mode == LOCAL_AUTO) ? "fast path" : "user specified";
      utils::logmesg(lmp, "Fix atom/swap: using LOCAL energy calculation ({})\n", mode_str);
    }
  } else {
    if (comm->me == 0)
      utils::logmesg(lmp, "Fix atom/swap: using FULL energy calculation\n");
  }
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo swaps
------------------------------------------------------------------------- */

void FixAtomSwap::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  
  neighbor->build(1);

  energy_stored = energy_full();

  if (use_local_energy) {
    if (local_energy_cache == nullptr || atom->nmax > max_affected) {
      memory->destroy(local_energy_cache);
      memory->destroy(local_energy_new);
      memory->destroy(is_affected);
      memory->destroy(affected_list);
      max_affected = atom->nmax;
      memory->create(local_energy_cache, max_affected, "atom/swap:local_energy_cache");
      memory->create(local_energy_new, max_affected, "atom/swap:local_energy_new");
      memory->create(is_affected, max_affected, "atom/swap:is_affected");
      memory->create(affected_list, max_affected, "atom/swap:affected_list");
      for (int i = 0; i < max_affected; i++) is_affected[i] = 0;
    }
    
    // Calculate average neighbors for hybrid heuristic
    if (atom->nlocal > 0 && list != nullptr) {
      long total_neighs = 0;
      for (int i = 0; i < list->inum; i++) total_neighs += list->numneigh[i];
      avg_neighs = (double)total_neighs / list->inum;
    } else {
      avg_neighs = 0.0;
    }

    int nlocal = atom->nlocal;
    if (list != nullptr && force->pair != nullptr) {
      for (int i = 0; i < nlocal; i++) {
        local_energy_cache[i] = force->pair->compute_atomic_energy(i, list);
      }
    }
  }

  // attempt Ncycle atom swaps

  int nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
  }

  // udpate MC stats

  nswap_attempts += ncycles;
  nswap_successes += nsuccess;

  next_reneighbor = update->ntimestep + nevery;

  mc_active = 0;

  // if visualization support is enabled, age vizatoms and remove expired ones
  if (vizsteps > 0) {
    std::vector<tagint> eraseme;
    for (const auto &[key, data] : vizatoms) {
      int idx = atom->map(key);
      if ((idx < 0) || (data.first < 0)) {
        eraseme.push_back(key);
        continue;
      }
      vizatoms[key] = std::make_pair(data.first - nevery, data.second);
    }
    for (const auto &key : eraseme) vizatoms.erase(key);
  }
}

/* ----------------------------------------------------------------------
   attempt a semd-grand swap of a single atom
   compare before/after energy and accept/reject the swap
   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  // pick a random atom and perform swap

  int itype, jtype, jswaptype;
  int i = pick_semi_grand_atom();
  if (i >= 0) {
    jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
    jtype = type_list[jswaptype];
    itype = atom->type[i];
    while (itype == jtype) {
      jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
      jtype = type_list[jswaptype];
    }
    atom->type[i] = jtype;
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy

  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  int success = 0;
  if (i >= 0)
    if (random_unequal->uniform() <
        exp(beta * (energy_before - energy_after + mu[jtype] - mu[itype])))
      success = 1;

  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  // swap accepted, return 1

  if (success_all) {
    update_semi_grand_atoms_list();
    energy_stored = energy_after;
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
        // record atom for which the type was swapped and store the old type
        if (vizsteps > 0) {
          vizatoms[atom->tag[i]] = std::make_pair(vizsteps,itype);
        }
      }
    }
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped atom
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  if (i >= 0) atom->type[i] = itype;
  if (force->kspace) force->kspace->qsum_qsq();

  return 0;
}

/* ----------------------------------------------------------------------
   attempt a swap of a pair of atoms
   compare before/after energy and accept/reject the swap
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_swap()
{
  // check if we have enough atoms for a group swap
  if ((niswap < group_size) || (njswap < group_size)) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  // pick group_size atoms of each type
  pick_iswap_group(local_swap_igroup, group_size);
  pick_jswap_group(local_swap_jgroup, group_size);

  int itype = type_list[0];
  int jtype = type_list[1];

  // --- LOCAL ENERGY FAST PATH ---

  if (use_local_energy) {
    // Hybrid Heuristic:
    // If nlocal is small (e.g. < 2 * avg_neighbors), the O(N_neigh^2) local delta 
    // is slower than O(N_local * N_neigh) full recalculation.
    // However, we still want to avoid Allreduce on tags.
    
    bool use_full_recalc = (atom->nlocal < 2.0 * avg_neighs);
    // Force local path for debugging/testing if needed, or stick to heuristic
    
    if (use_full_recalc) {
       // --- OPTIMIZED FULL PATH (Small N_local) ---
       // 1. Apply swap locally (using cached global tags)
       // 2. Sync ghosts (Forward comm)
       // 3. Recalculate full local energy
       // 4. Global Allreduce delta
       
       // Pre-calculate energy of current config (should match energy_stored/nprocs roughly?)
       // Actually, we need delta. 
       // E_before_local = sum(local_energy_cache)
       double e_before_local = 0.0;
       for(int i=0; i<atom->nlocal; i++) e_before_local += local_energy_cache[i];

       // Temporarily apply swap
       int swapped_indices[2*group_size];
       int n_swapped = 0;
       
       // Resolve indices (same as Accept block)
        std::vector<int> gi, gj;
        if (niswap > 0) {
            while ((int)gi.size() < group_size) {
            int candidate = static_cast<int>(niswap * random_equal->uniform());
            bool duplicate = false;
            for (int idx : gi) if (idx == candidate) { duplicate = true; break; }
            if (!duplicate) gi.push_back(candidate);
            }
        }
        if (njswap > 0) {
            while ((int)gj.size() < group_size) {
            int candidate = static_cast<int>(njswap * random_equal->uniform());
            bool duplicate = false;
            for (int idx : gj) if (idx == candidate) { duplicate = true; break; }
            if (!duplicate) gj.push_back(candidate);
            }
        }

       for (int n = 0; n < group_size; n++) {
        int idx_i = atom->map(all_iswap_tags[gi[n]]);
        int idx_j = atom->map(all_jswap_tags[gj[n]]);
        // Apply swap
        if (idx_i >= 0 && idx_i < atom->nlocal) {
            atom->type[idx_i] = jtype;
            if (atom->q_flag) atom->q[idx_i] = qtype[1];
            if (atom->rmass != nullptr) atom->rmass[idx_i] = mtype[1];
            swapped_indices[n_swapped++] = idx_i;
        }
        if (idx_j >= 0 && idx_j < atom->nlocal) {
            atom->type[idx_j] = itype;
            if (atom->q_flag) atom->q[idx_j] = qtype[0];
            if (atom->rmass != nullptr) atom->rmass[idx_j] = mtype[0];
            swapped_indices[n_swapped++] = idx_j;
        }
       }
       
       // Sync ghosts
       comm->forward_comm(this);
       
       // Compute new energy
       double e_after_local = 0.0;
       double *e_atom_ptr = local_energy_new; // reuse array
       for (int i=0; i<atom->nlocal; i++) {
           e_atom_ptr[i] = force->pair->compute_atomic_energy(i, list);
           e_after_local += e_atom_ptr[i];
       }
       
       double delta_local = e_after_local - e_before_local;
       double delta;
       MPI_Allreduce(&delta_local, &delta, 1, MPI_DOUBLE, MPI_SUM, world);
       
       if (random_equal->uniform() < exp(-beta * delta)) {
           // ACCEPTED
           // Update energy cache with new values
           for(int i=0; i<atom->nlocal; i++) local_energy_cache[i] = e_atom_ptr[i];
           
           // Velocities
           if (ke_flag) {
               for (int n = 0; n < group_size; n++) {
                   int idx_i = atom->map(all_iswap_tags[gi[n]]);
                   int idx_j = atom->map(all_jswap_tags[gj[n]]);
                   if (idx_i >= 0 && idx_i < atom->nlocal) {
                       atom->v[idx_i][0] *= sqrt_mass_ratio[itype][jtype];
                       atom->v[idx_i][1] *= sqrt_mass_ratio[itype][jtype];
                       atom->v[idx_i][2] *= sqrt_mass_ratio[itype][jtype];
                   }
                   if (idx_j >= 0 && idx_j < atom->nlocal) {
                       atom->v[idx_j][0] *= sqrt_mass_ratio[jtype][itype];
                       atom->v[idx_j][1] *= sqrt_mass_ratio[jtype][itype];
                       atom->v[idx_j][2] *= sqrt_mass_ratio[jtype][itype];
                   }
               }
           }
           energy_stored += delta;
           return 1;
       } else {
           // REJECTED
           // Revert types
           for(int i=0; i<n_swapped; i++) {
               int idx = swapped_indices[i];
               // If type was jtype, it means it was originally itype (so we set it back to itype)
               // Simple logic: if type == jtype, set to itype. if type == itype, set to jtype.
               if (atom->type[idx] == jtype) {
                   atom->type[idx] = itype;
                   if (atom->q_flag) atom->q[idx] = qtype[0];
                   if (atom->rmass != nullptr) atom->rmass[idx] = mtype[0];
               } else {
                   atom->type[idx] = jtype;
                   if (atom->q_flag) atom->q[idx] = qtype[1];
                   if (atom->rmass != nullptr) atom->rmass[idx] = mtype[1];
               }
           }
           comm->forward_comm(this);
           return 0;
       }

    } else {
        // --- EXISTING LOCAL DELTA PATH (Large N_local) ---
        double delta_local = energy_local_delta();
        double delta;
        MPI_Allreduce(&delta_local, &delta, 1, MPI_DOUBLE, MPI_SUM, world);

        if (random_equal->uniform() < exp(-beta * delta)) {
        // accepted: apply swap permanently on all procs
        // Update local energy cache
        for (int i = 0; i < n_affected_count; i++) {
            int idx = affected_list[i];
            local_energy_cache[idx] = local_energy_new[idx];
        }

        // Sync and pick global indices (already in pick_iswap_group, but need them here too)
        // Actually, we can just regenerate them since random_equal is synced
        std::vector<int> gi, gj;
        if (niswap > 0) {
            while ((int)gi.size() < group_size) {
            int candidate = static_cast<int>(niswap * random_equal->uniform());
            bool duplicate = false;
            for (int idx : gi) if (idx == candidate) { duplicate = true; break; }
            if (!duplicate) gi.push_back(candidate);
            }
        }
        if (njswap > 0) {
            while ((int)gj.size() < group_size) {
            int candidate = static_cast<int>(njswap * random_equal->uniform());
            bool duplicate = false;
            for (int idx : gj) if (idx == candidate) { duplicate = true; break; }
            if (!duplicate) gj.push_back(candidate);
            }
        }

        // Apply locally
        for (int n = 0; n < group_size; n++) {
            int idx_i = atom->map(all_iswap_tags[gi[n]]);
            int idx_j = atom->map(all_jswap_tags[gj[n]]);
            
            if (idx_i >= 0 && idx_i < atom->nlocal) {
            atom->type[idx_i] = jtype;
            if (atom->q_flag) atom->q[idx_i] = qtype[1];
            if (atom->rmass != nullptr) atom->rmass[idx_i] = mtype[1];
            if (ke_flag) {
                atom->v[idx_i][0] *= sqrt_mass_ratio[itype][jtype];
                atom->v[idx_i][1] *= sqrt_mass_ratio[itype][jtype];
                atom->v[idx_i][2] *= sqrt_mass_ratio[itype][jtype];
            }
            }
            if (idx_j >= 0 && idx_j < atom->nlocal) {
            atom->type[idx_j] = itype;
            if (atom->q_flag) atom->q[idx_j] = qtype[0];
            if (atom->rmass != nullptr) atom->rmass[idx_j] = mtype[0];
            if (ke_flag) {
                atom->v[idx_j][0] *= sqrt_mass_ratio[jtype][itype];
                atom->v[idx_j][1] *= sqrt_mass_ratio[jtype][itype];
                atom->v[idx_j][2] *= sqrt_mass_ratio[jtype][itype];
            }
            }
        }

        update_swap_atoms_list();
        energy_stored += delta;
        return 1;
        }
        return 0;
    }
  }

  // --- FULL ENERGY PATH (fallback) ---

  // swap all N pairs of atoms
  for (int n = 0; n < group_size; n++) {
    int i = local_swap_igroup[n];
    int j = local_swap_jgroup[n];

    if (i >= 0) {
      atom->type[i] = jtype;
      if (atom->q_flag) atom->q[i] = qtype[1];
      if (atom->rmass != nullptr) atom->rmass[i] = mtype[1];
    }
    if (j >= 0) {
      atom->type[j] = itype;
      if (atom->q_flag) atom->q[j] = qtype[0];
      if (atom->rmass != nullptr) atom->rmass[j] = mtype[0];
    }
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy

  double energy_after = energy_full();

  // swap accepted, return 1
  // if ke_flag, rescale atom velocities

  if (random_equal->uniform() < exp(beta * (energy_before - energy_after))) {
    update_swap_atoms_list();
    if (ke_flag) {
      for (int n = 0; n < group_size; n++) {
        int i = local_swap_igroup[n];
        int j = local_swap_jgroup[n];

        if (i >= 0) {
          atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
          atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
          atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
        }
        if (j >= 0) {
          atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
          atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
          atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
        }

        // record atoms for which the type was swapped and store the old types
        if (vizsteps > 0) {
          if (i >= 0) vizatoms[atom->tag[i]] = std::make_pair(vizsteps, jtype);
          if (j >= 0) vizatoms[atom->tag[j]] = std::make_pair(vizsteps, itype);
        }
      }
    }
    energy_stored = energy_after;
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped itype & jtype atoms
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  for (int n = 0; n < group_size; n++) {
    int i = local_swap_igroup[n];
    int j = local_swap_jgroup[n];

    if (i >= 0) {
      atom->type[i] = type_list[0];
      if (atom->q_flag) atom->q[i] = qtype[0];
      if (atom->rmass != nullptr) atom->rmass[i] = mtype[0];
    }
    if (j >= 0) {
      atom->type[j] = type_list[1];
      if (atom->q_flag) atom->q[j] = qtype[1];
      if (atom->rmass != nullptr) atom->rmass[j] = mtype[1];
    }
  }

  return 0;
}

/* ----------------------------------------------------------------------
   compute local energy change using affected-set algorithm
   returns delta_E = E_after - E_before for the affected set of atoms
   the swap is applied temporarily and then reverted before returning
------------------------------------------------------------------------- */

double FixAtomSwap::energy_local_delta()
{
  int nlocal = atom->nlocal;
  
  if (nlocal + atom->nghost > max_affected) {
    memory->destroy(is_affected);
    memory->destroy(affected_list);
    memory->destroy(local_energy_new);
    max_affected = atom->nmax;
    memory->create(is_affected, max_affected, "atom/swap:is_affected");
    memory->create(affected_list, max_affected, "atom/swap:affected_list");
    memory->create(local_energy_new, max_affected, "atom/swap:local_energy_new");
    for (int i = 0; i < max_affected; i++) is_affected[i] = 0;
  }

  int total_to_sync = (semi_grand_flag) ? 1 : 2 * group_size;
  std::vector<tagint> swapped_tags(total_to_sync, 0);
  std::vector<tagint> local_tags(total_to_sync, 0);
  
  if (!semi_grand_flag) {
    for (int n = 0; n < group_size; n++) {
      if (local_swap_igroup[n] >= 0) local_tags[n] = atom->tag[local_swap_igroup[n]];
      if (local_swap_jgroup[n] >= 0) local_tags[group_size + n] = atom->tag[local_swap_jgroup[n]];
    }
  }

  MPI_Allreduce(local_tags.data(), swapped_tags.data(), total_to_sync, MPI_LMP_TAGINT, MPI_MAX, world);

  std::vector<int> local_indices(total_to_sync);
  for (int n = 0; n < total_to_sync; n++) {
     local_indices[n] = atom->map(swapped_tags[n]);
  }

  int naffected = 0;
  for (int n = 0; n < total_to_sync; n++) {
    int idx = local_indices[n];
    if (idx < 0 || idx >= nlocal) continue;

    if (!is_affected[idx]) {
      is_affected[idx] = 1;
      affected_list[naffected++] = idx;
    }
    
    if (list == nullptr) continue;
    if (list->firstneigh == nullptr) continue;

    int *jlist = list->firstneigh[idx];
    int jnum = list->numneigh[idx];
    
    for (int jj = 0; jj < jnum; jj++) {
      int nbr = jlist[jj] & NEIGHMASK;
      if (nbr < nlocal && !is_affected[nbr]) {
        is_affected[nbr] = 1;
        affected_list[naffected++] = nbr;
      }
    }
  }

  n_affected_count = naffected;
  double e_before = 0.0;
  for (int i = 0; i < naffected; i++) e_before += local_energy_cache[affected_list[i]];

  int itype = type_list[0];
  int jtype = type_list[1];
  for (int n = 0; n < total_to_sync; n++) {
    int idx = local_indices[n];
    if (idx >= 0 && idx < nlocal) atom->type[idx] = (n < group_size) ? jtype : itype;
  }

  double e_after = 0.0;
  for (int i = 0; i < naffected; i++) {
    int atom_idx = affected_list[i];
    local_energy_new[atom_idx] = force->pair->compute_atomic_energy(atom_idx, list);
    e_after += local_energy_new[atom_idx];
  }

  for (int n = 0; n < total_to_sync; n++) {
    int idx = local_indices[n];
    if (idx >= 0 && idx < nlocal) atom->type[idx] = (n < group_size) ? itype : jtype;
  }
  
  for (int i = 0; i < naffected; i++) is_affected[affected_list[i]] = 0;
  return e_after - e_before;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixAtomSwap::energy_full()
{
  if (!c_pe) return 0.0;
  
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_semi_grand_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(nswap * random_equal->uniform());
  if ((iwhichglobal >= nswap_before) && (iwhichglobal < nswap_before + nswap_local)) {
    int iwhichlocal = iwhichglobal - nswap_before;
    i = local_swap_atom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_i_swap_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
  if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_j_swap_atom()
{
  int j = -1;
  int jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
  if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local)) {
    int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
  }

  return j;
}

/* ----------------------------------------------------------------------
   pick N atoms of type 1 for group swap (without replacement)
   selected indices stored in 'selected' array
   each proc determines which atoms from the global selection belong to it
------------------------------------------------------------------------- */

void FixAtomSwap::pick_iswap_group(int *selected, int nselect)
{
  for (int n = 0; n < nselect; n++) selected[n] = -1;
  if (niswap == 0) return;

  std::vector<int> global_indices;
  global_indices.reserve(nselect);
  
  while ((int)global_indices.size() < nselect) {
    int candidate = static_cast<int>(niswap * random_equal->uniform());
    bool duplicate = false;
    for (int idx : global_indices) {
      if (idx == candidate) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) global_indices.push_back(candidate);
  }

  for (int n = 0; n < nselect; n++) {
    tagint target_tag = all_iswap_tags[global_indices[n]];
    selected[n] = atom->map(target_tag);
    // only keep if local (atom->map returns local index or -1)
    if (selected[n] >= atom->nlocal) selected[n] = -1;
  }
}

/* ----------------------------------------------------------------------
   pick N atoms of type 2 for group swap (without replacement)
   selected indices stored in 'selected' array
   each proc determines which atoms from the global selection belong to it
------------------------------------------------------------------------- */

void FixAtomSwap::pick_jswap_group(int *selected, int nselect)
{
  for (int n = 0; n < nselect; n++) selected[n] = -1;
  if (njswap == 0) return;

  std::vector<int> global_indices;
  global_indices.reserve(nselect);
  
  while ((int)global_indices.size() < nselect) {
    int candidate = static_cast<int>(njswap * random_equal->uniform());
    bool duplicate = false;
    for (int idx : global_indices) {
      if (idx == candidate) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) global_indices.push_back(candidate);
  }

  for (int n = 0; n < nselect; n++) {
    tagint target_tag = all_jswap_tags[global_indices[n]];
    selected[n] = atom->map(target_tag);
    if (selected[n] >= atom->nlocal) selected[n] = -1;
  }
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_semi_grand_atoms_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_atom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_atom_list, atom_swap_nmax, "MCSWAP:local_swap_atom_list");
  }

  nswap_local = 0;

  if (region) {
    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          int itype = atom->type[i];
          int iswaptype;
          for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
            if (itype == type_list[iswaptype]) break;
          if (iswaptype == nswaptypes) continue;
          local_swap_atom_list[nswap_local] = i;
          nswap_local++;
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        int itype = atom->type[i];
        int iswaptype;
        for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
          if (itype == type_list[iswaptype]) break;
        if (iswaptype == nswaptypes) continue;
        local_swap_atom_list[nswap_local] = i;
        nswap_local++;
      }
    }
  }

  MPI_Allreduce(&nswap_local, &nswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&nswap_local, &nswap_before, 1, MPI_INT, MPI_SUM, world);
  nswap_before -= nswap_local;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_swap_atoms_list()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_iatom_list);
    memory->destroy(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_iatom_list, atom_swap_nmax, "MCSWAP:local_swap_iatom_list");
    memory->create(local_swap_jatom_list, atom_swap_nmax, "MCSWAP:local_swap_jatom_list");
  }

  niswap_local = 0;
  njswap_local = 0;

  if (region) {

    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[0]) {
            local_swap_iatom_list[niswap_local] = i;
            niswap_local++;
          } else if (type[i] == type_list[1]) {
            local_swap_jatom_list[njswap_local] = i;
            njswap_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        if (type[i] == type_list[0]) {
          local_swap_iatom_list[niswap_local] = i;
          niswap_local++;
        } else if (type[i] == type_list[1]) {
          local_swap_jatom_list[njswap_local] = i;
          njswap_local++;
        }
      }
    }
  }

  MPI_Allreduce(&niswap_local, &niswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&niswap_local, &niswap_before, 1, MPI_INT, MPI_SUM, world);
  niswap_before -= niswap_local;

  MPI_Allreduce(&njswap_local, &njswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&njswap_local, &njswap_before, 1, MPI_INT, MPI_SUM, world);
  njswap_before -= njswap_local;

  if (use_local_energy) {
    int nprocs = comm->nprocs;
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);

    // Sync type-1 tags
    MPI_Allgather(&niswap_local, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
    displs[0] = 0;
    for (int i = 1; i < nprocs; i++) displs[i] = displs[i-1] + recvcounts[i-1];
    all_iswap_tags.assign(niswap, 0);
    std::vector<tagint> local_itags(niswap_local);
    for (int i = 0; i < niswap_local; i++) local_itags[i] = atom->tag[local_swap_iatom_list[i]];
    MPI_Allgatherv(local_itags.data(), niswap_local, MPI_LMP_TAGINT,
                   all_iswap_tags.data(), recvcounts.data(), displs.data(), MPI_LMP_TAGINT, world);

    // Sync type-2 tags
    MPI_Allgather(&njswap_local, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
    displs[0] = 0;
    for (int i = 1; i < nprocs; i++) displs[i] = displs[i-1] + recvcounts[i-1];
    all_jswap_tags.assign(njswap, 0);
    std::vector<tagint> local_jtags(njswap_local);
    for (int i = 0; i < njswap_local; i++) local_jtags[i] = atom->tag[local_swap_jatom_list[i]];
    MPI_Allgatherv(local_jtags.data(), njswap_local, MPI_LMP_TAGINT,
                   all_jswap_tags.data(), recvcounts.data(), displs.data(), MPI_LMP_TAGINT, world);
  }
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int>(buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++) type[i] = static_cast<int>(buf[m++]);
  }
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */

double FixAtomSwap::compute_vector(int n)
{
  if (n == 0) return nswap_attempts;
  if (n == 1) return nswap_successes;
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixAtomSwap::memory_usage()
{
  double bytes = (double) atom_swap_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixAtomSwap::write_restart(FILE *fp)
{
  int n = 0;
  double list[6];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = nswap_attempts;
  list[n++] = nswap_successes;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), n, fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixAtomSwap::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;

  seed = static_cast<int>(list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int>(list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nswap_attempts = static_cast<int>(list[n++]);
  nswap_successes = static_cast<int>(list[n++]);

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR, "Must not reset timestep when restarting fix atom/swap");
}

/* ----------------------------------------------------------------------
   extract variable which stores whether MC is active or not
     active = MC moves are taking place
     not active = normal MD is taking place
------------------------------------------------------------------------- */

void *FixAtomSwap::extract(const char *name, int &dim)
{
  if (strcmp(name,"mc_active") == 0) {
    dim = 0;
    return (void *) &mc_active;
  }
  return nullptr;
}

/* ----------------------------------------------------------------------
   provide graphics information to dump image to render spheres
   at the location of atoms that were involved in a reaction
------------------------------------------------------------------------- */

int FixAtomSwap::image(int *&objs, double **&parms)
{
  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, Error::NOLASTLINE,
               "Cannot use fix atom/swap in dump image without an atom map");

  memory->destroy(imgobjs);
  memory->destroy(imgparms);

  int numobjs = vizatoms.size();
  int n = 0;
  if (numobjs > 0) {
    memory->create(imgobjs, numobjs, "atom/swap:imgobjs");
    memory->create(imgparms, numobjs, 5, "atom/swap:imgparms");

    int idx;
    const auto *const *const x = atom->x;
    for (const auto &[key, data] : vizatoms) {
      idx = atom->map(key);
      if (idx < 0) continue;
      imgobjs[n] = Graphics::SPHERE;
      imgparms[n][0] = data.second; // use stored pre-swap atom type
      imgparms[n][1] = x[idx][0];
      imgparms[n][2] = x[idx][1];
      imgparms[n][3] = x[idx][2];
      imgparms[n][4] = 0.0;     // radius is set with fflag2 in dump image
      ++n;
    }
  }
  objs = imgobjs;
  parms = imgparms;
  return n;
}

/* ----------------------------------------------------------------------
   capture the neighbor list requested by the fix
------------------------------------------------------------------------- */

void FixAtomSwap::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}
