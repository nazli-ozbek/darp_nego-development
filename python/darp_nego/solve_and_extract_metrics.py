#!/usr/bin/env python3

"""Solve DARP problems from a company_cases.json and extract core solution metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import re
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

from darp_nego.darp.basic_darp import BasicDARPVehicle, BasicDARPClient, BasicDARPProblem
from darp_nego.logger.darp_routing_logger import DARPRoutingLogger
from visualize_darp_metrics import visualize_metrics


def _load_company_cases(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def _natural_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _sorted_case_names(
    data: Dict[str, Any],
    case_id: str | None,
    case_ids: List[str] | None,
) -> List[str]:
    if case_id:
        return [case_id] if case_id in data else []
    if case_ids:
        names = [cid for cid in case_ids if cid in data]
        return sorted(names, key=_natural_key)
    names = [k for k in data.keys() if k not in {"time_matrix", "coordinates"}]
    return sorted(names, key=_natural_key)


def _build_problem(company_name: str, company_data: Dict[str, Any], time_matrix: List[List[float]],
                   coordinates: Dict[str, Any]) -> BasicDARPProblem:
    try:
        agent_id = int(company_name.split("_")[1])
    except Exception:
        agent_id = 0

    vehicles = [BasicDARPVehicle.from_json(v) for v in company_data.get("vehicles", [])]
    clients = []
    for c in company_data.get("clients", []):
        start_key = str(c.get("start_location"))
        end_key = str(c.get("end_location"))
        start_coords = coordinates.get(start_key)
        end_coords = coordinates.get(end_key)
        if start_coords is None or end_coords is None:
            raise KeyError(f"Missing coordinates for client {c.get('client_id')} ({start_key}->{end_key})")

        clients.append(
            BasicDARPClient(
                c["client_id"],
                c["start_location"],
                c["end_location"],
                c["early_pickup"],
                c["late_pickup"],
                c["early_drop_off"],
                c["late_drop_off"],
                c["volume"],
                start_coords,
                end_coords,
            )
        )

    problem = BasicDARPProblem(agent_id, vehicles=vehicles, clients=clients)
    problem.road_network = time_matrix
    problem.coordinates = coordinates
    return problem


def _compute_basic_metrics(route_summary: Dict[str, Any]) -> Dict[str, Any]:
    total_time = route_summary.get("total_time", 0)
    total_waiting_time = route_summary.get("total_waiting_time", 0)
    vehicles_used = route_summary.get("vehicles_used", 0)
    total_vehicles = route_summary.get("total_vehicles", 0)
    served_clients = route_summary.get("served_clients", 0)

    total_delays = route_summary.get("total_delays", 0)

    late_arrivals = route_summary.get("late_arrivals", [])
    total_lates = [l.get("late_by", 0) for l in late_arrivals]

    avg_total_delay = float(np.mean(total_lates)) if total_lates else 0.0
    max_total_delay = float(max(total_lates)) if total_lates else 0.0

    route_times = [r.get("duration", 0) for r in route_summary.get("routes", []) if r.get("stops", 0) > 0]
    max_route_time = float(max(route_times)) if route_times else 0.0
    avg_route_time = float(np.mean(route_times)) if route_times else 0.0

    total_capacity = 0
    total_load = 0
    for route in route_summary.get("routes", []):
        if route.get("stops", 0) > 0:
            total_capacity += route.get("capacity", 0)
            total_load += route.get("max_load", 0)

    capacity_utilization = (total_load / total_capacity) if total_capacity > 0 else 0.0
    time_utilization = ((total_time - total_waiting_time) / total_time) if total_time > 0 else 0.0
    vehicle_utilization = (vehicles_used / total_vehicles) if total_vehicles > 0 else 0.0

    return {
        "total_time": total_time,
        "total_waiting_time": total_waiting_time,
        "vehicles_used": vehicles_used,
        "vehicle_utilization": vehicle_utilization,
        "served_clients": served_clients,
        "total_delays": total_delays,
        "avg_total_delay": avg_total_delay,
        "max_total_delay": max_total_delay,
        "max_route_time": max_route_time,
        "avg_route_time": avg_route_time,
        "capacity_utilization": capacity_utilization,
        "time_utilization": time_utilization,
    }


def solve_and_extract_metrics(
    input_path: str,
    output_dir: str,
    case_id: str | None = None,
    case_ids: List[str] | None = None,
) -> str:
    data = _load_company_cases(input_path)
    output_dir = os.path.normpath(output_dir)
    if os.path.basename(output_dir) == "darp_metrics":
        metrics_dir = output_dir
    else:
        metrics_dir = os.path.join(output_dir, "darp_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    logger = DARPRoutingLogger(log_dir=metrics_dir, session_id="darp_metrics", create_dir=False)

    results: List[Dict[str, Any]] = []

    for case_name in _sorted_case_names(data, case_id=case_id, case_ids=case_ids):
        case_data = data[case_name]

        time_matrix = case_data.get("time_matrix", [])
        coordinates = case_data.get("coordinates", {})
        companies = case_data.get("companies", {})

        for company_name, company_data in companies.items():
            problem = _build_problem(company_name, company_data, time_matrix, coordinates)

            start = time.perf_counter()
            try:
                solution_cost = problem.solve_problem()
                solve_status = "ok" if solution_cost is not None else "infeasible"
            except Exception as exc:
                solution_cost = None
                solve_status = f"error: {exc.__class__.__name__}"
            solve_time_sec = time.perf_counter() - start

            route_summary: Dict[str, Any] = {}
            if solve_status == "ok":
                route_summary = logger._extract_route_data(company_name, problem)

            metrics = _compute_basic_metrics(route_summary) if route_summary else {}

            results.append({
                "case_name": case_name,
                "company_name": company_name,
                "num_vehicles": len(company_data.get("vehicles", [])),
                "num_clients": len(company_data.get("clients", [])),
                "solve_status": solve_status,
                "solve_time_sec": round(solve_time_sec, 4),
                "solution_cost": solution_cost if solution_cost is not None else "",
                **metrics,
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(metrics_dir, f"{base_name}_darp_metrics_{timestamp}.csv")

    fieldnames: List[str] = []
    for row in results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    visuals_dir = os.path.join(metrics_dir, "visuals", f"plots_{timestamp}")
    visualize_metrics(output_path, visuals_dir)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve DARP scenarios and extract core metrics")
    parser.add_argument("--input", required=True, help="Path to company_cases.json")
    parser.add_argument("--out", required=True, help="Output directory for metrics JSON and plots")
    parser.add_argument("--case-id", default=None, help="Optional case id to process")
    parser.add_argument("--case-ids", default=None, help="Comma-separated list of case ids to process")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Process at most this many cases (deterministic natural order).",
    )
    args = parser.parse_args()

    case_ids = [c.strip() for c in args.case_ids.split(",")] if args.case_ids else None
    if case_ids:
        case_ids = sorted(case_ids, key=_natural_key)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be >= 1")
        if case_ids is None and args.case_id is None:
            data = _load_company_cases(args.input)
            case_keys = [k for k, v in data.items() if isinstance(v, dict) and "companies" in v]
            case_ids = sorted(case_keys, key=_natural_key)[: args.max_cases]
    output_path = solve_and_extract_metrics(args.input, args.out, case_id=args.case_id, case_ids=case_ids)
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
