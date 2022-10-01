/* CLOMP Benchmark to measure OpenMP overheads with typical usage situations.
 * Also measures cache effects and memory bandwidth affects on performance.
 * Provides memory layout options that may affect performance.
 * Removes earlier variations that didn't provide expected insight.
 * Version 1.2 Drastically reduces serial overhead for high part count
 *             inputs, allowing reasonable messages with large thread counts.
 *             Also added -DWITH_MPI to measure OpenMP overheads when run
 *             with multiple MPI tasks per node, defaults to no MPI.
 *             Reduced digits reported after decimal point for most measurments.
 *             Last output line prints CORAL RFP data for easy importing.
 *             Changes by John Gyllenhaal at LLNL 12/23/13
 * Version 1.1 calculates error bounds (but not yet used) and dramatically
 *             reduces chance of underflows in math.  John Gyllenhaal 6/16/10
 * Version 1 (first public release) of the CLOMP benchmark.
 * Written by John Gyllenhaal and Greg Bronevetsky at LLNL 5/25/07
 * Based on version 0.2 (4/24/07) and version 0.1 (12/14/07) by John and Greg.
 ******************************************************************************
COPYRIGHT AND LICENSE

Copyright (c) 2007, The Regents of the University of California.
Produced at the Lawrence Livermore National Laboratory
Written by John Gyllenhaal (gyllen@llnl.gov)
and Greg Bronevetsky (bronevetsky1@llnl.gov)
UCRL-CODE-233406.
All rights reserved.

This file is part of CLOMP.
For details, see www.llnl.gov/asc/sequoia/benchmarks

Redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the disclaimer (as noted below) in
  the documentation and/or other materials provided with the distribution.

* Neither the name of the UC/LLNL nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY
OF CALIFORNIA, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ADDITIONAL BSD NOTICE

1. This notice is required to be provided under our contract with the
   U.S. Department of Energy (DOE). This work was produced at the
   University of California, Lawrence Livermore National Laboratory
   under Contract No. W-7405-ENG-48 with the DOE.

2. Neither the United States Government nor the University of California
   nor any of their employees, makes any warranty, express or implied,
   or assumes any liability or responsibility for the accuracy, completeness,
   or usefulness of any information, apparatus, product, or process disclosed,
   or represents that its use would not infringe privately-owned rights.

3. Also, reference herein to any specific commercial products, process,
   or services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or the University of California.
   The views and opinions of authors expressed herein do not necessarily
   state or reflect those of the United States Government or the University
   of California, and shall not be used for advertising or product
   endorsement purposes.
******************************************************************************/

extern "C"
{
#include <runtime/runtime.h>
}

#include "deref_scope.hpp"
#include "dataframe_vector.hpp"
#include "stats.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

using namespace far_memory;
using namespace std;

/* Command line parameters, see usage info (initially -1 for sanity check)*/
long CLOMP_numThreads = -2;   /* > 0 or -1 valid */
long CLOMP_allocThreads = -2; /* > 0 or -1 valid */
long CLOMP_numParts = -1;     /* > 0 valid */
long CLOMP_zonesPerPart = -1; /* > 0 valid */
long CLOMP_flopScale = -1;    /* > 0 valid, 1 nominal */
long CLOMP_timeScale = -1;    /* > 0 valid, 100 nominal */
long CLOMP_zoneSize = -1;     /* > 0 valid, (sizeof(Zone) true min)*/
char *CLOMP_exe_name = NULL;  /* Points to argv[0] */

/* Save actual argument for summary at end */
long CLOMP_inputAllocThreads = 1;

/* Simple Zone data structure */
class CZone
{
public:
    long m_iZoneId;
    long m_iPartId;
    double value;
    SharedPtr<CZone> *m_pNextZone = nullptr;
};

/* Part data structure */
class CPart
{
public:
    long m_iPartId;
    long m_iZoneCount;
    long m_iUpdate_count;

    SharedPtr<CZone> *m_pFirstZone = nullptr;
    SharedPtr<CZone> *m_pLastZone = nullptr;

    double m_dDeposit_ratio;
    double m_dResidue;
    double m_dExpected_first_value; /* Used to check results */
    double m_dExpected_residue;     /* Used to check results */
};

/* Part array working on (now array of Part pointers)*/
// far_memory::Array<SharedPtr<CPart>, CLOMP_numParts> *g_paPartArray;

FarMemManager *far_mem_manager;
std::vector<SharedPtr<CPart>> g_paPartArray;
std::vector<std::vector<SharedPtr<CZone>>> g_paZoneArray;

/* Used to avoid dividing by numParts */
double CLOMP_partRatio = 0.0;

/* Number of iterations to do approximately 0.5 to 2 second of work on a
 * fast new machine (deterministic calculated using heuristics).   */
long CLOMP_num_iterations = 0.0;

/* Used to check m_dResidue of checksum */
double CLOMP_max_residue = 0.0;

/* Used to calc_deposit's return value without having to check
 * every part's m_dResidue.  This allows more parts to be used and
 * thus higher thread counts.
 */
double CLOMP_residue_ratio_part0 = 0.0;

/* Used to determine if a calculation difference is a rounding error or
 * the result of the compiler, runtime, or test doing something bad
 * (i.e., races, bad barriers, etc.).
 * The loose error bound is calculated to be slightly smaller than
 * the smallest update of any zone.  This should catch races preventing
 * an update from happenning.
 */
double CLOMP_error_bound = 0.0;

/* The tight error bound is calculated to be the smallest difference between
 * two adjacent zone updates.   This should detect swapped updates (which
 * probably is not possible).   May be tighter than needed.
 */
double CLOMP_tightest_error_bound = 0.0;

void print_usage()
{
    fprintf(stderr,
            "Usage: clomp numThreads allocThreads numParts \\\n"
            "           zonesPerPart zoneSize flopScale timeScale\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "New in Version 1.2: Compile with -DWITH_MPI to generate clomp_mpi\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  numThreads: Number of OpenMP threads to use (-1 for system default)\n");
    fprintf(stderr, "  allocThreads: #threads when allocating data (-1 for numThreads)\n");
    fprintf(stderr, "  numParts: Number of independent pieces of work (loop iterations)\n");
    fprintf(stderr, "  zonesPerPart: Number of zones in the first part (3 flops/zone/part)\n");
    fprintf(stderr, "  zoneSize: Bytes in zone, only first ~32 used (512 nominal, >= 32 valid)\n");
    fprintf(stderr, "  flopScale: Scales flops/zone to increase memory reuse (1 nominal, >=1 Valid)\n");
    fprintf(stderr, "  timeScale: Scales target time per test (10-100 nominal, 1-10000 Valid)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Some interesting testcases (last number controls run time):\n");
    fprintf(stderr, "           Target input:    clomp 16 1 16 400 32 1 100\n");
    fprintf(stderr, "   Target/NUMA friendly:    clomp 16 -1 16 400 32 1 100\n");
    fprintf(stderr, "    Weak Scaling Target:    clomp N -1 N 400 32 1 100\n");
    fprintf(stderr, "      Weak Scaling Huge:    clomp N -1 N 6400 32 1 100\n");
    fprintf(stderr, "  Strong Scaling Target:    clomp -1 -1 1024 10 32 1 100\n");
    fprintf(stderr, "        Mem-bound input:    clomp N 1 N 640000 32 1 100\n");
    fprintf(stderr, "Mem-bound/NUMA friendly:    clomp N -1 N 640000 32 1 100\n");
    fprintf(stderr, "  MPI/OMP Hybrid Target:    (mpirun -np M) clomp_mpi 16 1 16 400 32 1 100\n");
}
/* Convert parm_val to a positive long value.  Punts if negative or not
 * completely translated
 */
long convert_to_positive_long(const char *parm_name, const char *parm_val)
{
    long val;
    char *endPtr;

    /* Sanity check */
    if ((parm_name == nullptr) || (parm_val == nullptr))
    {
        printf("Error in convert_to_positive_long: Passed nullptr pointers!\n");
        exit(1);
    }

    /* Convert string to long */
    val = strtol(parm_val, &endPtr, 0);

    /* Make sure everything in string was converted */
    if (endPtr[0] != 0)
    {
        fprintf(stderr, "Error converting '%s' parameter value '%s' to long at '%s'!\n",
                parm_name, parm_val, endPtr);
        exit(1);
    }

    /* Make sure final value > 0 except for numThreads and allocThreads,
     * which also takes -1
     */
    if ((strcmp(parm_name, "numThreads") == 0) ||
        (strcmp(parm_name, "allocThreads") == 0))
    {
        if ((val < 1) && (val != -1))
        {
            fprintf(stderr, "Invalid value %ld for parameter %s, must be > 0 or -1!\n",
                    val, parm_name);

            print_usage();
            exit(1);
        }
    }
    /* Make sure final value > 0 (except for numThreads checked above) */
    else if (val < 1)
    {
        fprintf(stderr, "Invalid value %ld for parameter %s, must be > 0\n",
                val, parm_name);

        print_usage();
        exit(1);
    }

    return (val);
}

void update_part(long iPartId, double incoming_deposit)
{
    double m_dDeposit_ratio, remaining_deposit, deposit;
    {
        CPart *part_raw_ptr = nullptr;
        {
            DerefScope vScope;
            part_raw_ptr = g_paPartArray[iPartId].deref_mut(vScope);
        }

        /* Update count of updates for this part (for error checking)
         * Just part 0's count will be zeroed regularly.   Others may wrap.
         */
        part_raw_ptr->m_iUpdate_count++;

        /* Get the m_dDeposit_ratio from part*/

        m_dDeposit_ratio = part_raw_ptr->m_dDeposit_ratio;

        /* Initially, the remaining_deposit is the incoming deposit */
        remaining_deposit = incoming_deposit;

        /* If have the most common case (original case) where CLOMP_flopScale = 1,
         * use a specialized loop to get best performance for this important case.
         * (Since the faster the loop, the more OpenMP overhead matters.)
         */
        if (CLOMP_flopScale == 1)
        {
            /* Run through each zone, depositing 'm_dDeposit_ratio' part of the
             * remaining_deposit in the zone and carrying the rest to the remaining
             * zones
             */

            for (auto zone = part_raw_ptr->m_pFirstZone; zone != nullptr;)
            {
                DerefScope vScope;
                auto zone_raw_ptr = zone->deref_mut(vScope);
                /* Calculate the deposit for this zone */
                deposit = remaining_deposit * m_dDeposit_ratio;

                /* Add deposit to the zone's value */
                zone_raw_ptr->value += deposit;

                /* Remove deposit from the remaining_deposit */
                remaining_deposit -= deposit;

                // loop interation
                zone = zone_raw_ptr->m_pNextZone;
            }
        }

        /* Otherwise, if CLOMP_flopScale != 1, use inner loop version */
        else
        {
            /* Run through each zone, depositing 'm_dDeposit_ratio' part of the
             * remaining_deposit in the zone and carrying the rest to the remaining
             * zones
             */

            for (auto zone = part_raw_ptr->m_pFirstZone; zone != nullptr;)
            {
                DerefScope vScope;
                auto zone_raw_ptr = zone->deref_mut(vScope);

                /* Allow scaling of the flops per double loaded, so that you
                 * can get expensive iterations without blowing the cache.
                 */
                for (long scale_count = 0; scale_count < CLOMP_flopScale; scale_count++)
                {
                    /* Calculate the deposit for this SharedPtr<Zone> */
                    deposit = remaining_deposit * m_dDeposit_ratio;

                    /* Add deposit to the zone's value */
                    zone_raw_ptr->value += deposit;

                    /* Remove deposit from the remaining_deposit */
                    remaining_deposit -= deposit;
                }

                // loop interation
                zone = zone_raw_ptr->m_pNextZone;
            }
        }

        /* Put the left over deposit in the Part's m_dResidue field */
        part_raw_ptr->m_dResidue = remaining_deposit;
    }
}

/* Resets parts to initial state and warms up cache */
void reinitialize_parts()
{
    /* Reset all the zone values to 0.0 and the part m_dResidue to 0.0 */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        CPart *part_raw_ptr = nullptr;
        {
            DerefScope vInternalScope;
            part_raw_ptr = g_paPartArray[pidx].deref_mut(vInternalScope);
        }

        for (auto zone = part_raw_ptr->m_pFirstZone; zone != nullptr;)
        {
            DerefScope vScope;
            auto zone_raw_ptr = zone->deref_mut(vScope);

            /* Reset zone's value to 0 */
            zone_raw_ptr->value = 0.0;

            // loop interation
            zone = zone_raw_ptr->m_pNextZone;
        }

        /* Reset m_dResidue */
        part_raw_ptr->m_dResidue = 0.0;

        /* Reset update count */
        part_raw_ptr->m_iUpdate_count = 0;
    }

    /* Scan through zones and add zero to each zone to warm up cache*/
    /* Also sets each zone m_iUpdate_count to 1, which sanity check wants */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, 0.0);
    }
}

/* Helper routine to print out timestamp when test starting */
void print_start_message(const char *desc)
{
    time_t starttime;
    char startdate[50]; /* Must be > 26 characters */

    /* Get time/data starting run*/
    std::time(&starttime);
    ctime_r(&starttime, startdate);

    /* Print out start message, startdate includes newline */
    printf("%13s  Started: %s", desc, startdate);

    /* Include number of threads if not "Serial Ref"or "calc_deposit"
     */
    if (strcmp(desc, "calc_deposit") != 0)
    {
        if (strcmp(desc, "Serial Ref") != 0)
        {
            /* Print out how many threads we are using */
            printf("%13s #Threads: %d\n", desc, 1);
        }
        else
        {
            /* Otherwise, print out N/A for Serial Ref */
            printf("%13s #Threads: N/A\n", desc);
        }
    }
}

/* Helper routine that puts time of day into passed ts structure */
void get_timestamp(struct timeval *ts)
{
    if (gettimeofday(ts, nullptr) != 0)
    {
        fprintf(stderr, "Unable to get time of day, exiting\n");
        exit(1);
    }
}

/* Helper routine that prints a line of the benchmark pseudocode
 * prefixed with the description
 */
void print_pseudocode(const char *desc, const char *pseudocode)
{
    printf("%13s:| %s\n", desc, pseudocode);
}

/* Prints and returns run time in seconds (as double).
 * If base_seconds > 0.0, prints speedup numbers also
 */
double print_timestats(const char *desc, struct timeval *start_ts,
                       struct timeval *end_ts, double base_seconds,
                       double bestcase_seconds)
{
    double seconds;
    char used_desc[100] = "";

    /* Calculate run time */
    seconds = ((double)end_ts->tv_sec + ((double)end_ts->tv_usec * 1e-6)) -
              ((double)start_ts->tv_sec + ((double)start_ts->tv_usec * 1e-6));

    /* Print out overall runtime */
    printf("%13s  Runtime: %.3f (wallclock, in seconds%s)\n", desc, seconds,
           used_desc);

    /* Print runtime per subcycle in microseconds */
    printf("%13s  us/Loop: %.2f (wallclock, in microseconds)\n", desc,
           (seconds * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));

    /* If valid base time passed in and seconds valid, print speedups
     * (or slowdowns, if necessary)
     */
    if ((base_seconds > 0.0) && (seconds > 0.0))
    {
        if (base_seconds > seconds)
        {
            printf("%13s  Speedup: %.1f\n", desc, base_seconds / seconds);
        }
        else
        {
            printf("%13s  Speedup: %.1f (%.1fX slowdown)\n", desc,
                   base_seconds / seconds, seconds / base_seconds);
        }
    }

    if ((bestcase_seconds > 0.0) && (seconds > 0.0))
    {
        printf("%13s Efficacy: %.1f%% (of bestcase %.2f us/Loop)\n", desc,
               (bestcase_seconds / seconds) * 100.0,
               (bestcase_seconds * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));

        printf("%13s Overhead: %.2f (versus bestcase, in us/Loop)\n",
               desc, ((seconds - bestcase_seconds) * 1000000.0) / ((double)CLOMP_num_iterations * 10.0));
    }

    printf("---------------------\n");

    return (seconds);
}

/* Check data for consistency and print out a one line data stat summary
 * that should help tell if data is all right  (total should be number
 * of iterations run).
 */
void print_data_stats(const char *desc)
{
    double value_sum, residue_sum, last_value, dtotal;
    int is_reference, error_count;

    /* Initialize value and m_dResidue sums to zero */
    value_sum = 0.0;
    residue_sum = 0.0;

    /* Use "Serial Ref" as the reference calculation and check all
     * values against that.  Since the same code is used to calculate
     * all these values, we should not have any issues with reordering
     * calculations causing small differences.
     */
    if (strcmp(desc, "Serial Ref") == 0)
        is_reference = 1;
    else
        is_reference = 0;

    /* Initialize count of check errors */
    error_count = 0;

    /* Scan through each part, check that values decrease monotonically
     * and sum up all the values.  Also check that the part m_dResidue and
     * the part's first zone's value are the expected value.
     */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        /* If have reference calculation, grab the first zone's value
         * and the part m_dResidue for comparison later
         */

        CPart *part_raw_ptr = nullptr;
        if (is_reference)
        {
            DerefScope vScope;
            part_raw_ptr = g_paPartArray[pidx].deref_mut(vScope);
            part_raw_ptr->m_dExpected_first_value = part_raw_ptr->m_pFirstZone->deref_mut(vScope)->value;
            part_raw_ptr->m_dExpected_residue = part_raw_ptr->m_dResidue;
        }

        /* Otherwise, make sure this part matches the expected values
         * from the reference calculation.  Since using exactly the same
         * code to calculate these values (for all variations) and the
         * values are calculated in the same order (independent of number
         * of threads), I expect these values to be exactly the same.
         * If not true in the future, may have to put bounds on this.
         */
        else
        {
            auto firstZoneValue = 0.0;
            {
                DerefScope vScope;
                firstZoneValue = part_raw_ptr->m_pFirstZone->deref_mut(vScope)->value;
            }

            /* Check that first zone's value is what is expected */
            if (part_raw_ptr->m_dExpected_first_value != firstZoneValue)
            {
                DerefScope vScope;
                error_count++;
                fprintf(stderr,
                        "%s check failure: part %i first zone value (%g) != reference value (%g)!\n",
                        desc, (int)pidx, part_raw_ptr->m_pFirstZone->deref_mut(vScope)->value,
                        part_raw_ptr->m_dExpected_first_value);
            }
            if (part_raw_ptr->m_dExpected_residue != part_raw_ptr->m_dResidue)
            {
                DerefScope vScope;
                error_count++;
                printf("%s check failure: part %i m_dResidue (%g) != reference m_dResidue (%g)!\n",
                       desc, (int)pidx, part_raw_ptr->m_dResidue, part_raw_ptr->m_dExpected_residue);
            }
        }

        {
            DerefScope vScope;
            last_value = part_raw_ptr->m_pFirstZone->deref_mut(vScope)->value;
        }
        /* Use first zone's value as initial last_value */

        /* Scan through zones checking that values decrease monotonically */
        for (auto zone = part_raw_ptr->m_pFirstZone; zone != nullptr;)
        {
            DerefScope vScope;
            auto zone_raw_ptr = zone->deref_mut(vScope);
            if (zone_raw_ptr->value > last_value)
            {
                printf("*** %s check failure (part %i zone %i): "
                       "previous (%g) < current (%g)!\n",
                       desc, (int)zone_raw_ptr->m_iPartId,
                       (int)zone_raw_ptr->m_iZoneId, last_value, zone_raw_ptr->value);
                error_count++;
            }

            /* Sum up values */
            value_sum += zone_raw_ptr->value;

            /* This value now is last_value */
            last_value = zone_raw_ptr->value;

            // loop interation
            zone = zone_raw_ptr->m_pNextZone;
        }

        /* Sum up part m_dResidue's */
        residue_sum += part_raw_ptr->m_dResidue;
    }

    /* Calculate the total of value_sum + residue_sum.  This should
     * equal the number of subcycles run (due to construction of benchmark)
     */
    dtotal = value_sum + residue_sum;

    /* Sanity check, use reasonable bounds before alert */
    if (((dtotal + 0.00001) < ((double)CLOMP_num_iterations * 10.0)) ||
        ((dtotal - 0.00001) > ((double)CLOMP_num_iterations * 10.0)))
    {
        printf("*** %s check failure:  Total (%-.15g) != Expected (%.15g)\n",
               desc, dtotal, ((double)CLOMP_num_iterations * 10.0));
        error_count++;
    }

    /* Sanity check, m_dResidue must be within reasonable bounds */
    if ((residue_sum < 0.0) || (residue_sum > (CLOMP_max_residue + 0.000001)))
    {
        printf("*** %s check failure: Residue (%-.15g) outside bounds 0 - %.15g\n",
               desc, residue_sum, CLOMP_max_residue);
        error_count++;
    }

    /* Make sure part 0's update count is exactly one.  This detects
     * illegal optimization of calc_deposit().
     */
    CPart *zero_index_part_raw_ptr = nullptr;
    {
        DerefScope vScope;
        zero_index_part_raw_ptr = g_paPartArray[0].deref_mut(vScope);
    }

    if (zero_index_part_raw_ptr->m_iUpdate_count != 1)
    {
        printf("Error in calc_deposit: Part updated %i times since last calc_deposit!\n",
               (int)zero_index_part_raw_ptr->m_iUpdate_count);
        printf("Benchmark designed to have calc_deposit called exactly once per update!\n");
        printf("Critical error: Exiting...\n");
        exit(1);
    }

    if (error_count > 0)
    {
        printf("ERROR: %i check failures detected in '%s' data. Exiting...\n",
               error_count, desc);
        exit(1);
    }

    /* Print out check text so results can be visually inspected */
    printf("%13s Checksum: Sum=%-8.8g Residue=%-8.8g Total=%-.9g\n", desc, value_sum, residue_sum, dtotal);
}

/* Calculates the amount to deposit in each part this subcycle.
 * Based on the m_dResidue of all parts from last subcycle.
 * Normally, this would be done with some sort of MPI exchange of
 * data but here we are just using info from local zones.
 *
 * Should be used by all subcycles to calculate amount to deposit.
 *
 * Cannot be moved past subcycle loops without getting wrong answer.
 */
double calc_deposit()
{
    double m_dResidue, deposit;

    /* Sanity check, make sure residues have be updated since last calculation
     * This code cannot be pulled out of loops or above loops!
     * Only check/update part 0 (other counts will just continue and may wrap)
     */
    CPart *zero_index_part_raw_ptr = nullptr;
    {
        DerefScope vScope;
        zero_index_part_raw_ptr = g_paPartArray[0].deref_mut(vScope);
    }
    if (zero_index_part_raw_ptr->m_iUpdate_count != 1)
    {
        printf("Error in calc_deposit: Part updated %i times since last call!\n",
               (int)zero_index_part_raw_ptr->m_iUpdate_count);
        printf("Benchmark designed to have calc_deposit called exactly once per update!\n");
        printf("Critical error: Exiting...\n");
        exit(1);
    }

    /* Mark that we are using the updated info, so we can detect if this
     * code has been moved illegally.
     */
    zero_index_part_raw_ptr->m_iUpdate_count = 0;

    /* Calculate m_dResidue from previous subcycle (normally this is done
     * with an MPI data exchange to other domains, but emulate here).
     */
    /* This method doesn't scale well past 64 parts with target input.
     * So using new method below to allow large part counts which
     * enables larger thread counts.
     *  -JCG 17Dec2013
     *
     * m_dResidue = 0.0;
     * for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
     * {
     *   m_dResidue += g_paPartArray[pidx]->m_dResidue;
     * }
     */

    /* Use direct calculation of deposit based on part 0's m_dResidue to
     * allow large part counts to be used.   Still want some data to
     * be used from calculation to prevent undesired optimizations.
     * Hopefully using part 0's data will be enough -JCG 17Dec2013
     */
    m_dResidue = zero_index_part_raw_ptr->m_dResidue * CLOMP_residue_ratio_part0;

    /* Calculate deposit for this subcycle based on m_dResidue and part ratio */
    deposit = (1.0 + m_dResidue) * CLOMP_partRatio;

    /* Return the amount to deposit in each part this subcycle */
    return (deposit);
}

/* Do only the non-threadable calc_deposit() calls (without the update_part
 * loops) so that part of the benchmark can be timed separately.
 * Since we are not touching all the zone data, we run faster due to
 * having to cache much less data.  Unfortunately, this may make this
 * test run faster than when surrounded by the real code.
 */
void do_calc_deposit_only()
{
    long iteration, subcycle;

    /* Do all the iterations */
    for (iteration = 0; iteration < CLOMP_num_iterations; iteration++)
    {
        /* 10 subcycles to every iteration, calc_deposit call in each one */
        for (subcycle = 0; subcycle < 10; subcycle++)
        {

            CPart *zero_index_part_raw_ptr = nullptr;
            {
                DerefScope vScope;
                zero_index_part_raw_ptr = g_paPartArray[0].deref_mut(vScope);
            }

            /* Fool calc_deposit sanity checks for this timing measurement */
            zero_index_part_raw_ptr->m_iUpdate_count = 1;

            /* Calc value, write into first zone's value, in order
             * to prevent compiler optimizing away
             */
            {

                auto zero_firstZone_far_mem_ptr = zero_index_part_raw_ptr->m_pFirstZone;
                DerefScope vScope;
                auto zero_firstZone_raw_ptr = zero_firstZone_far_mem_ptr->deref_mut(vScope);
                zero_firstZone_raw_ptr->value = calc_deposit();
            }
        }
    }
}

/* Do module one's work serially (contains 1 subcycle) */
void serial_ref_module1()
{
    double deposit;
    /* ---------------- SUBCYCLE 1 OF 1 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }
}

/* Do module two's work serially (contains 2 subcycles) */
void serial_ref_module2()
{
    double deposit;
    /* ---------------- SUBCYCLE 1 OF 2 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        // DerefScope vScope;
        // auto pPartArryIndexPtr = (&(g_paPartArray->at_mut(vScope, pidx)));
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 2 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }
}

/* Do module three's work serially (contains 3 subcycles) */
void serial_ref_module3()
{
    double deposit;
    /* ---------------- SUBCYCLE 1 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 3 OF 3 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }
}

/* Do module four's work serially (contains 4 subcycles) */
void serial_ref_module4()
{
    double deposit;

    /* ---------------- SUBCYCLE 1 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 2 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 3 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }

    /* ---------------- SUBCYCLE 4 OF 4 ----------------- */

    /* Calculate deposit for this subcycle based on last subcycle's m_dResidue */
    deposit = calc_deposit();

    /* Scan through zones and add appropriate deposit to each SharedPtr<Zone> */
    for (long pidx = 0; pidx < CLOMP_numParts; pidx++)
    {
        update_part(pidx, deposit);
    }
}

/* Do one cycle (10 subcycles) serially, no OpenMP */
void serial_ref_cycle()
{
    /* Emulate calls to 4 different packages, do 10 subcycles total */
    serial_ref_module1();
    serial_ref_module2();
    serial_ref_module3();
    serial_ref_module4();
}

/* Do all the cycles (10 subcycles/cycle) serially, no OpenMP */
void do_serial_ref_version()
{
    long iteration;

    /* Do the specified number of iterations */
    for (iteration = 0; iteration < CLOMP_num_iterations; iteration++)
        serial_ref_cycle();
}

/* Helper, add part passed in to the g_paPartArray at m_iPartId and initialize
 * it.  The separate routine is used to make it easy to allocate parts
 * with various strategies (such as each thread allocating it's parts).
 * The g_paPartArray has to be allocated by one thread but it is not
 * modified during the run.
 */
void addPart(long iPartId)
{
    /* Sanity check, make sure m_iPartId valid */
    if ((iPartId < 0) || (iPartId >= CLOMP_numParts))
    {
        fprintf(stderr, "addPart error: m_iPartId (%i) out of bounds!\n", (int)iPartId);
        exit(1);
    }

    DerefScope vScope;
    auto pPartObject = g_paPartArray[iPartId].deref_mut(vScope);

    /* Set m_iPartId */
    pPartObject->m_iPartId = iPartId;

    /* Set zone count for part (now fixed, used to be variable */
    pPartObject->m_iZoneCount = CLOMP_zonesPerPart;

    /* Updated June 2010 by John Gyllenhaal to pick a deposit ratio
     * for this part that keeps the math from underflowing and
     * makes it possible to come up with a sane error bounds.
     * This math was picked experimentally to come up with
     * relatively large error bounds for the 'interesting testcase' inputs.
     *
     * This is part of an effort to separate rounding error due
     * to inlining update_part() (and optimizing different ways) from
     * incorrect results due to races or bad hardware.
     *
     * The older m_dDeposit_ratio only really worked well for 100 zones.
     */
    pPartObject->m_dDeposit_ratio = ((double)((1.5 * (double)CLOMP_numParts) + iPartId)) /
                                    ((double)(CLOMP_zonesPerPart * CLOMP_numParts));

    /* Initially no m_dResidue from previous passes */
    pPartObject->m_dResidue = 0.0;

    /* Initially, no zones attached to Part */
    pPartObject->m_pFirstZone = nullptr;
    pPartObject->m_pLastZone = nullptr;

    /* Initially, don't know expected values (used for checking */
    pPartObject->m_dExpected_first_value = -1.0;
    pPartObject->m_dExpected_residue = -1.0;
}

/* Appends zone to the part identified by m_iPartId.   Done in separate routine
 * to facilitate the zones being allocated in various ways (such as by
 * different threads, randomly, etc.)
 */
void addZone(long iPartId, long iZoneID)
{
    SharedPtr<CZone> *current_part_lastZone_far_mem_ptr = nullptr;
    {
        DerefScope vOuterScope;
        current_part_lastZone_far_mem_ptr = g_paPartArray[iPartId].deref_mut(vOuterScope)->m_pLastZone;
    }

    /* If not existing zones, place at head of list */
    if (current_part_lastZone_far_mem_ptr == nullptr)
    {
        DerefScope vInternalScope;
        /* Give first zone a zoneId of 1 */
        g_paZoneArray[iPartId][iZoneID].deref_mut(vInternalScope)->m_iZoneId = 1;

        /* First and last zone */
        g_paPartArray[iPartId].deref_mut(vInternalScope)->m_pFirstZone = &g_paZoneArray[iPartId][iZoneID];
        g_paPartArray[iPartId].deref_mut(vInternalScope)->m_pLastZone = &g_paZoneArray[iPartId][iZoneID];
    }
    else /* Otherwise, put after last zone */
    {
        DerefScope vInternalScope;
        g_paZoneArray[iPartId][iZoneID].deref_mut(vInternalScope)->m_iZoneId = g_paPartArray[iPartId].deref_mut(vInternalScope)->m_pLastZone->deref_mut(vInternalScope)->m_iZoneId + 1;
        g_paPartArray[iPartId].deref_mut(vInternalScope)->m_pLastZone->deref_mut(vInternalScope)->m_pNextZone = &g_paZoneArray[iPartId][iZoneID];
        g_paPartArray[iPartId].deref_mut(vInternalScope)->m_pLastZone = &g_paZoneArray[iPartId][iZoneID];
    }

    {
        DerefScope vInternalScope;
        auto zone_raw_ptr = g_paZoneArray[iPartId][iZoneID].deref_mut(vInternalScope);

        /* Inialized the rest of the zone fields */
        zone_raw_ptr->m_pNextZone = nullptr;
        zone_raw_ptr->m_iPartId = g_paPartArray[iPartId].deref_mut(vInternalScope)->m_iPartId;
        zone_raw_ptr->value = 0.0;
    }
}

/*
 * --------------------------------------------------------------------
 * Main driver
 * --------------------------------------------------------------------
 */

int argc;
std::string ip_addr_port;
void _main(void *arg)
{
    char **argv = static_cast<char **>(arg);
    char hostname[200];
    time_t starttime;
    char startdate[50]; /* Must be > 26 characters */
    double totalZoneCount;
    double deposit, percent_residue, part_deposit_bound;
    double deposit_diff_bound;
    double diterations;
    struct timeval calc_deposit_start_ts, calc_deposit_end_ts;
    double calc_deposit_seconds;
    struct timeval serial_ref_start_ts, serial_ref_end_ts;
    double serial_ref_seconds;

    constexpr static uint64_t kCacheSize = (128ULL << 20);
    constexpr static uint64_t kFarMemSize = (4ULL << 30);
    constexpr static uint32_t kNumGCThreads = 12;
    // constexpr static uint32_t kNumEntries = (8ULL << 20); // So the array size is larger than the local cache size.
    constexpr static uint32_t kNumConnections = 300;

    auto raddr = helpers::str_to_netaddr(ip_addr_port);
    auto manager = std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new TCPDevice(raddr, kNumConnections, kFarMemSize)));
    far_mem_manager = manager.get();

    /* Get executable name by pointing to argv[0] */
    CLOMP_exe_name = argv[0];

    printf("CORAL Benchmark Version 1.2\n");

    /* Print usage if not 7 arguments */
    if (argc != 8)
    {
        print_usage();
        exit(1);
    }

    /* Get hostname running on */
    if (gethostname(hostname, sizeof(hostname)) != 0)
        strcpy(hostname, "(Unknown host)");

    /* Get date starting benchmark on */
    std::time(&starttime);
    ctime_r(&starttime, startdate);

    /* Read in command line args (all must be positive ints) */
    CLOMP_numThreads = convert_to_positive_long("numThreads", argv[1]);
    CLOMP_allocThreads = convert_to_positive_long("numThreads", argv[2]);
    CLOMP_numParts = convert_to_positive_long("numParts", argv[3]);
    CLOMP_zonesPerPart = convert_to_positive_long("zonesPerPart", argv[4]);
    CLOMP_zoneSize = convert_to_positive_long("zoneSize", argv[5]);
    CLOMP_flopScale = convert_to_positive_long("flopScale", argv[6]);
    CLOMP_timeScale = convert_to_positive_long("timeScale", argv[7]);

    /* Zone size cannot be less than sizeof(Zone), force to be valid */
    if ((long unsigned int)CLOMP_zoneSize < sizeof(CZone))
    {
        printf("***Forcing zoneSize (%ld specified) to minimum zone size %ld\n\n",
               CLOMP_zoneSize, (long)sizeof(CZone));
        CLOMP_zoneSize = sizeof(CZone);
    }

    /* Print out command line arguments as passed in */
    printf("       Invocation:");
    for (auto aidx = 0; aidx < argc; aidx++)
    {
        printf(" %s", argv[aidx]);
    }
    printf("\n");

    /* Print out command line arguments read in */
    printf("         Hostname: %s\n", hostname);
    printf("       Start time: %s", startdate); /* startdate has newline */
    printf("       Executable: %s\n", CLOMP_exe_name);
    if (CLOMP_numThreads == -1)
    {
        /* Set CLOMP_numThreads to system default if -1 */
        CLOMP_numThreads = 1;

        /* Print out system default for threads */
        printf("      numThreads: %d (using system default)\n", (int)CLOMP_numThreads);
    }
    else
    {
        printf("      numThreads: %ld\n", CLOMP_numThreads);
    }

    /* Save actual argument for summary at end */
    CLOMP_inputAllocThreads = CLOMP_allocThreads;

    /* If -1, use numThreads for alloc threads */
    if ((int)CLOMP_allocThreads == -1)
    {
        /* Use numThreads for alloc threads if -1 */
        CLOMP_allocThreads = CLOMP_numThreads;

        /* Print out number of alloc threads to use */
        printf("    allocThreads: %ld (using numThreads)\n", CLOMP_allocThreads);
    }
    else
    {
        /* Print out number of alloc threads to use */
        printf("    allocThreads: %ld\n", CLOMP_allocThreads);
    }

    printf("        numParts: %ld\n", CLOMP_numParts);
    printf("    zonesPerPart: %ld\n", CLOMP_zonesPerPart);
    printf("       flopScale: %ld\n", CLOMP_flopScale);
    printf("       timeScale: %ld\n", CLOMP_timeScale);
    printf("        zoneSize: %ld\n", CLOMP_zoneSize);

    /* Allocate part pointer array */
    for (long int i = 0; i < CLOMP_numParts; i++)
    {
        auto part_far_mem_ptr = far_mem_manager->allocate_shared_ptr<CPart>();
        {
            DerefScope scope;
            auto raw_part_ptr = part_far_mem_ptr.deref_mut(scope);
            raw_part_ptr = new CPart();
        }
        g_paPartArray.emplace_back(std::move(part_far_mem_ptr));

        /* Allocate zone pointer array */
        std::vector<SharedPtr<CZone>> TempVect;
        for (long int j = 0; j < CLOMP_zonesPerPart; j++)
        {
            auto zone_far_mem_ptr = far_mem_manager->allocate_shared_ptr<CZone>();
            {
                DerefScope scope;
                auto raw_zone_ptr = zone_far_mem_ptr.deref_mut(scope);
                raw_zone_ptr = new CZone();
            }
            TempVect.emplace_back(zone_far_mem_ptr);
        }
        g_paZoneArray.emplace_back(TempVect);
    }

    /* Calculate 1/numParts to prevent divides in algorithm */
    CLOMP_partRatio = 1.0 / ((double)CLOMP_numParts);

    /* Ininitialize parts (with no zones initially).
     * Do allocations in thread (allocThreads may be set to 1 for allocate)
     * to allow potentially better memory layout for threads
     */

    for (long iPartId = 0; iPartId < CLOMP_numParts; iPartId++)
    {
        addPart(iPartId);
    }

    for (long iPartId = 0; iPartId < CLOMP_numParts; iPartId++)
    {
        for (long int zoneId = 0; zoneId < CLOMP_zonesPerPart; zoneId++)
        {
            addZone(iPartId, zoneId);
        }
    }

    /* Calculate the total number of zones */
    totalZoneCount = (double)CLOMP_numParts * (double)CLOMP_zonesPerPart;

    printf("   Zones per Part: %.0f\n", (double)CLOMP_zonesPerPart);
    printf("      Total Zones: %.0f\n", (double)totalZoneCount);
    printf("Memory (in bytes): %.0f\n", (double)(totalZoneCount * CLOMP_zoneSize) + (double)(sizeof(CPart) * CLOMP_numParts));

    /* Calculate a number of iterations that experimentally appears to
     * give between 0.05 and 1.3 seconds of work on a 2GHz Opteron.
     * The 0.05 for small inputs that fit in cache, 1.3 for others.
     * The minimum iterations is 1, so once 1 million zones is hit,
     * the iterations cannot drop any more to compensate.
     * Also factor in flop_scaling now so setting flop_scaling to 100
     * doesn't make benchmark run 100X longer.
     */
    diterations = ceil((((double)1000000) * ((double)CLOMP_timeScale)) /
                       ((double)totalZoneCount * (double)CLOMP_flopScale));

    /* Sanity check for very small zone counts */
    if (diterations > 2000000000.0)
    {
        printf("*** Forcing iterations from (%g) to 2 billion\n",
               diterations);
        diterations = 2000000000.0;
    }

    /* Convert double iterations to int for use */
    CLOMP_num_iterations = (long)diterations;

    printf("Scaled Iterations: %i\n", (int)CLOMP_num_iterations);
    printf("  Total Subcycles: %.0f\n",
           (double)CLOMP_num_iterations * (double)10.0);

    /* Calculate serially the percent m_dResidue left after one pass */
    percent_residue = 0.0;

    /* If we deposit 1.0/numParts, sum of residues becomes percent m_dResidue */
    deposit = CLOMP_partRatio;

    /* In order to have sane rounding error bounds, determine the minimum
     * amount deposited in any zone.   Initialize to deposit as a starting
     * point for the min calculation below.
     */
    CLOMP_error_bound = deposit;
    CLOMP_tightest_error_bound = deposit;

    /* Scan through zones and add deposit to each zone */
    for (long iPartId = 0; iPartId < CLOMP_numParts; iPartId++)
    {
        /* Do serial calculation of percent m_dResidue */

        update_part(iPartId, deposit);

        CPart *part_raw_ptr = nullptr;
        {
            DerefScope vScope;
            part_raw_ptr = g_paPartArray[iPartId].deref_mut(vScope);
        }
        percent_residue += part_raw_ptr->m_dResidue;

        /* The 'next' deposit is smaller than any actual deposit made in
         * this part.  So it is a good lower bound on deposit size.
         */
        part_deposit_bound = part_raw_ptr->m_dResidue * part_raw_ptr->m_dDeposit_ratio;

        /* If we use the smallest bound on any part, we should have the
         * biggest error bound that will not give false negatives on
         * bad computation.
         */
        if (CLOMP_error_bound > part_deposit_bound)
        {
#if 0
	    printf ("After part %i bound from %g to %g (deposit %g)\n", (int)m_iPartId,
		    CLOMP_error_bound, part_deposit_bound, part_raw_ptr->m_dDeposit_ratio);
#endif
            CLOMP_error_bound = part_deposit_bound;
        }
#if 0
	else
	{
	    printf ("After part %i bound stayed %g because %g (deposit %g)\n", (int)m_iPartId,
		    CLOMP_error_bound, part_deposit_bound, part_raw_ptr->m_dDeposit_ratio);
	}
#endif

        deposit_diff_bound = part_deposit_bound * part_raw_ptr->m_dDeposit_ratio;
        if (CLOMP_tightest_error_bound > deposit_diff_bound)
        {
#if 0
	    printf ("After part %i tightest bound from %g to %g (deposit %g)\n", (int)m_iPartId,
		    CLOMP_tightest_error_bound, deposit_diff_bound, part_raw_ptr->m_dDeposit_ratio);
#endif
            CLOMP_tightest_error_bound = deposit_diff_bound;
        }
#if 0
	else
	{
	    printf ("After part %i tightest bound stayed %g because %g (deposit %g)\n", (int)m_iPartId,
		    CLOMP_tightest_error_bound, deposit_diff_bound, part_raw_ptr->m_dDeposit_ratio);
	}
#endif
    }
    printf("Iteration Residue: %.6f%%\n", percent_residue * 100.0);
    printf("  Max Error bound: %-8.8g\n", CLOMP_error_bound);
    printf("Tight Error bound: %-8.8g\n", CLOMP_tightest_error_bound);

#if 0
    printf ("DEBUG ENDING NOW\n");
    exit (1);
#endif

    /* Calculate the expected converged m_dResidue (for infinite iterations)
     * If Y is the percent m_dResidue after one pass, then the converged
     * expected converged m_dResidue = (added_each_cycle * Y)/ (1-Y)
     */
    CLOMP_max_residue = (1.0 * percent_residue) / (1 - percent_residue);
    printf("      Max Residue: %-8.8g\n", CLOMP_max_residue);

    /* In order to deal with huge thread counts, we need to make calc_deposit
     * not depend on the number of parts.   We do this by quickly caculating
     * an correct m_dResidue that still relies on at least part 0's data
     * (in an attempt to prevent undesired compiler optimizations).
     * Found part 0's m_dResidue is directly proportional to entire m_dResidue,
     * so calculate the ratio to get correct value. -JCG 17Dec2013
     */
    {
        CPart *zero_index_part_raw_ptr = nullptr;
        {
            DerefScope vScope;
            zero_index_part_raw_ptr = g_paPartArray[0].deref_mut(vScope);
        }
        CLOMP_residue_ratio_part0 = percent_residue / zero_index_part_raw_ptr->m_dResidue;
    }

    /* Set the number of threads for the computation seciton to what user
     * specified. Because we are using alloc threads also, have to explicitly
     * set even if using system default
     */
    /* Print initial line bar separator */
    printf("---------------------\n");

    /* --------- Start calc_deposit() overhead measurement --------- */

    /* DON'T REINITIALIZE PARTS BEFORE do_calc_deposit_only TEST!
     * We want there to be non-zero data in residues.
     */

    /* Calculate just the overhead of the non-threadable calc_deposit() calls.
     * May get good cache effects here, so may be low estimate.
     */
    print_pseudocode("calc_deposit", "------ Start calc_deposit Pseudocode ------");
    print_pseudocode("calc_deposit", "/* Measure *only* non-threadable calc_deposit() overhead.*/");
    print_pseudocode("calc_deposit", "/* Expect this overhead to be negligible.*/");
    print_pseudocode("calc_deposit", "deposit = calc_deposit ();");
    print_pseudocode("calc_deposit", "------- End calc_deposit Pseudocode -------");
    print_start_message("calc_deposit");

    get_timestamp(&calc_deposit_start_ts);
    do_calc_deposit_only();
    get_timestamp(&calc_deposit_end_ts);
    calc_deposit_seconds = print_timestats("calc_deposit",
                                           &calc_deposit_start_ts,
                                           &calc_deposit_end_ts, -1.0, -1.0);

    /* --------- Start Serial Ref benchmark measurement --------- */

    /* Do one cycle outside timer loop to warm up code and (for OpenMP cases)
     * allow the OpenMP system to initialize (which can be expensive, skewing
     * the measurments for small runtimes */
    reinitialize_parts();
    serial_ref_cycle();

    /* Reinitialize parts and warm up cache by doing dummy update */
    reinitialize_parts();

    /* Do the serial version of calculation and measure time*/
    print_pseudocode("Serial Ref", "------ Start Serial Ref Pseudocode ------");
    print_pseudocode("Serial Ref", "/* Measure serial reference performance */");
    print_pseudocode("Serial Ref", "deposit = calc_deposit ();");
    print_pseudocode("Serial Ref", "for (pidx = 0; pidx < numParts; pidx++)");
    print_pseudocode("Serial Ref", "  update_part (partArray[pidx], deposit);");
    print_pseudocode("Serial Ref", "------- End Serial Ref Pseudocode -------");
    print_start_message("Serial Ref");

    get_timestamp(&serial_ref_start_ts);
    do_serial_ref_version();
    get_timestamp(&serial_ref_end_ts);

    /* Check data for consistency and print out data stats*/
    print_data_stats("Serial Ref");

    /* Print out serial time stats and capture time,
     * -1 prevents speedup numbers from printing out
     */
    serial_ref_seconds = print_timestats("Serial Ref", &serial_ref_start_ts,
                                         &serial_ref_end_ts, -1.0, -1.0);

    printf("----------- Comma-delimited summary ----------\n");
    printf("%s %ld %ld %ld %ld %ld %ld %ld, calc_deposit, Serial Ref\n",
           CLOMP_exe_name,
           CLOMP_numThreads,
           CLOMP_inputAllocThreads,
           CLOMP_numParts,
           CLOMP_zonesPerPart,
           CLOMP_zoneSize,
           CLOMP_flopScale,
           CLOMP_timeScale);

    printf("Runtime, %7.3f, %7.3f\n", calc_deposit_seconds, serial_ref_seconds);

#undef us_loop
#define us_loop(sec) (((sec * 1000000.0) / ((double)CLOMP_num_iterations * 10.0)))
    printf("us/Loop, %7.2f, %7.2f\n", us_loop(calc_deposit_seconds), us_loop(serial_ref_seconds));

#undef speedup
#define speedup(sec) ((serial_ref_seconds / sec))
    printf("Speedup,     N/A,     N/A, %7.1f\n", speedup(serial_ref_seconds));

    {
        printf("CORAL RFP, %ld %ld %ld %ld %ld %ld %ld, %.2f\n",
               CLOMP_numThreads,
               CLOMP_inputAllocThreads,
               CLOMP_numParts,
               CLOMP_zonesPerPart,
               CLOMP_zoneSize,
               CLOMP_flopScale,
               CLOMP_timeScale,
               us_loop(serial_ref_seconds));
    }

    for (long unsigned int iPartsIndex = 0; iPartsIndex < g_paPartArray.size(); iPartsIndex++)
    {
        g_paPartArray[iPartsIndex].free();
    }

    for (long unsigned int iPartsIndex = 0; iPartsIndex < g_paZoneArray.size(); iPartsIndex++)
    {
        for (long unsigned int iZoneIndex = 0; iZoneIndex < g_paZoneArray[iPartsIndex].size(); iZoneIndex++)
        {
            g_paZoneArray[iPartsIndex][iZoneIndex].free();
        }
    }

    cout << "_main() ended successfully.\n";
}

int main(int _argc, char *argv[])
{
    int ret;

    if (_argc < 3)
    {
        std::cerr << "usage: [cfg_file] [ip_addr:port]" << std::endl;
        return -EINVAL;
    }

    char conf_path[strlen(argv[1]) + 1];
    strcpy(conf_path, argv[1]);
    ip_addr_port.assign(argv[2]);
    for (int i = 3; i < _argc; i++)
    {
        argv[i - 2] = argv[i];
    }
    argc = _argc - 2;

    ret = runtime_init(conf_path, _main, argv);
    if (ret)
    {
        std::cerr << "failed to start runtime" << std::endl;
        return ret;
    }
    cout << "main() ended successfully.\n";
    return 0;
}