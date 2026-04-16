"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  getPhysicalFailureStudy,
  getPhysicalFailureStudyContext,
  listPhysicalFailureStudies,
  updatePhysicalFailureStudyEntry,
  type PhysicalFailureStudyContext,
  type PhysicalFailureStudyDetail,
  type PhysicalFailureStudyEntry,
  type PhysicalFailureStudyListItem,
} from "@/lib/api";

function formatPercent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function formatFrameIndex(value: number): string {
  return `f${value.toString().padStart(4, "0")}`;
}

function bucketLabel(entry: PhysicalFailureStudyEntry): string {
  return entry.final_bucket?.trim() || "(untagged)";
}

function detailsText(entry: PhysicalFailureStudyEntry): string {
  if (entry.notes?.trim()) return entry.notes.trim();
  if (entry.suggested_root_cause?.trim()) return entry.suggested_root_cause.trim();
  return "No notes";
}

function badgeVariantForBucket(
  bucket: string,
): "default" | "secondary" | "destructive" | "outline" {
  if (bucket === "(untagged)") return "outline";
  if (bucket.includes("eval") || bucket.includes("other")) return "secondary";
  if (bucket.includes("decoder")) return "destructive";
  return "default";
}

function matchesEntry(
  entry: PhysicalFailureStudyEntry,
  query: string,
  bucketFilter: string,
): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  const currentBucket = bucketLabel(entry);
  if (bucketFilter !== "all" && currentBucket !== bucketFilter) {
    return false;
  }
  if (!normalizedQuery) {
    return true;
  }

  const haystack = [
    entry.selected_index,
    entry.clip_filename,
    entry.annotation_id,
    entry.frame_index,
    entry.suggested_root_cause,
    entry.final_bucket,
    entry.notes,
    entry.best_legal_move_uci,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return haystack.includes(normalizedQuery);
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="mt-2 text-2xl font-semibold">{value}</div>
        {hint ? <div className="mt-1 text-xs text-muted-foreground">{hint}</div> : null}
      </CardContent>
    </Card>
  );
}

export default function PhysicalFailureStudyViewer() {
  const [studies, setStudies] = useState<PhysicalFailureStudyListItem[]>([]);
  const [selectedStudyPath, setSelectedStudyPath] = useState<string>("");
  const [study, setStudy] = useState<PhysicalFailureStudyDetail | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [context, setContext] = useState<PhysicalFailureStudyContext | null>(null);
  const [contextFrames, setContextFrames] = useState(10);
  const [currentFrameOffset, setCurrentFrameOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [bucketFilter, setBucketFilter] = useState("all");
  const [draftBucket, setDraftBucket] = useState("");
  const [draftNotes, setDraftNotes] = useState("");
  const [loadingStudies, setLoadingStudies] = useState(true);
  const [loadingStudy, setLoadingStudy] = useState(false);
  const [loadingContext, setLoadingContext] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    async function loadStudies() {
      try {
        setLoadingStudies(true);
        const response = await listPhysicalFailureStudies();
        setStudies(response.studies);
        setSelectedStudyPath((current) => {
          if (current && response.studies.some((study) => study.path === current)) {
            return current;
          }
          return response.studies[0]?.path || "";
        });
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to load failure studies");
      } finally {
        setLoadingStudies(false);
      }
    }

    void loadStudies();
  }, []);

  useEffect(() => {
    if (!selectedStudyPath) {
      setStudy(null);
      setSelectedIndex(null);
      return;
    }

    let cancelled = false;
    async function loadStudy() {
      try {
        setLoadingStudy(true);
        const response = await getPhysicalFailureStudy(selectedStudyPath);
        if (cancelled) return;
        setStudy(response);
        setSelectedIndex((current) => {
          if (current && response.entries.some((entry) => entry.selected_index === current)) {
            return current;
          }
          return response.entries[0]?.selected_index ?? null;
        });
      } catch (error) {
        if (cancelled) return;
        toast.error(error instanceof Error ? error.message : "Failed to load study");
        setStudy(null);
        setSelectedIndex(null);
      } finally {
        if (!cancelled) {
          setLoadingStudy(false);
        }
      }
    }

    void loadStudy();
    return () => {
      cancelled = true;
    };
  }, [selectedStudyPath]);

  useEffect(() => {
    if (!selectedStudyPath || selectedIndex == null) {
      setContext(null);
      return;
    }

    const currentSelectedIndex = selectedIndex;
    let cancelled = false;
    async function loadContext() {
      try {
        setLoadingContext(true);
        const response = await getPhysicalFailureStudyContext({
          path: selectedStudyPath,
          selected_index: currentSelectedIndex,
          context_frames: contextFrames,
        });
        if (cancelled) return;
        setContext(response);
        setCurrentFrameOffset(Math.max(0, response.frames.length - 1));
        setDraftBucket(response.entry.final_bucket ?? "");
        setDraftNotes(response.entry.notes ?? "");
      } catch (error) {
        if (cancelled) return;
        toast.error(error instanceof Error ? error.message : "Failed to load context");
        setContext(null);
      } finally {
        if (!cancelled) {
          setLoadingContext(false);
        }
      }
    }

    void loadContext();
    return () => {
      cancelled = true;
    };
  }, [contextFrames, selectedIndex, selectedStudyPath]);

  const bucketOptions = useMemo(() => {
    const values = new Set<string>(study?.bucket_options ?? []);
    for (const entry of study?.entries ?? []) {
      const value = entry.final_bucket?.trim();
      if (value) values.add(value);
    }
    return ["all", "(untagged)", ...Array.from(values)];
  }, [study]);

  const filteredEntries = useMemo(() => {
    return (study?.entries ?? []).filter((entry) => matchesEntry(entry, query, bucketFilter));
  }, [bucketFilter, query, study]);

  useEffect(() => {
    if (!filteredEntries.length) {
      return;
    }
    if (selectedIndex == null) {
      setSelectedIndex(filteredEntries[0]!.selected_index);
      return;
    }
    if (!filteredEntries.some((entry) => entry.selected_index === selectedIndex)) {
      setSelectedIndex(filteredEntries[0]!.selected_index);
    }
  }, [filteredEntries, selectedIndex]);

  const selectedEntry = useMemo(() => {
    return study?.entries.find((entry) => entry.selected_index === selectedIndex) ?? null;
  }, [selectedIndex, study]);

  const currentFrame =
    context && context.frames.length > 0
      ? context.frames[Math.min(currentFrameOffset, context.frames.length - 1)]
      : null;

  async function handleSave() {
    if (!study || selectedEntry == null) return;
    try {
      setSaving(true);
      const response = await updatePhysicalFailureStudyEntry({
        study_path: study.path,
        selected_index: selectedEntry.selected_index,
        final_bucket: draftBucket,
        notes: draftNotes,
      });

      setStudy((current) => {
        if (!current) return current;
        return {
          ...current,
          entries: current.entries.map((entry) =>
            entry.selected_index === response.selected_index
              ? {
                  ...entry,
                  final_bucket: response.final_bucket,
                  notes: response.notes,
                }
              : entry,
          ),
          bucket_counts: current.entries.reduce<Record<string, number>>((counts, entry) => {
            const bucket =
              entry.selected_index === response.selected_index
                ? response.final_bucket.trim() || "(untagged)"
                : bucketLabel(entry);
            counts[bucket] = (counts[bucket] ?? 0) + 1;
            return counts;
          }, {}),
        };
      });
      setContext((current) => {
        if (!current || current.entry.selected_index !== response.selected_index) {
          return current;
        }
        return {
          ...current,
          entry: {
            ...current.entry,
            final_bucket: response.final_bucket,
            notes: response.notes,
          },
        };
      });
      toast.success(`Saved bucket for failure ${response.selected_index}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to save failure bucket");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Physical Failure Study Viewer</CardTitle>
          <CardDescription>
            Browse tagged failure-study bundles, scrub the raw frames leading into each
            failure, and keep buckets/notes in sync with <code>manual_buckets.csv</code>.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 lg:grid-cols-[minmax(0,2fr)_160px_160px]">
            <label className="space-y-1 text-sm">
              <span className="font-medium">Study bundle</span>
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                disabled={loadingStudies || studies.length === 0}
                value={selectedStudyPath}
                onChange={(event) => setSelectedStudyPath(event.target.value)}
              >
                {studies.length === 0 ? (
                  <option value="">No failure studies found</option>
                ) : null}
                {studies.map((item) => (
                  <option key={item.path} value={item.path}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1 text-sm">
              <span className="font-medium">Leading frames</span>
              <input
                type="number"
                min={0}
                max={30}
                value={contextFrames}
                onChange={(event) =>
                  setContextFrames(Math.max(0, Math.min(30, Number(event.target.value) || 0)))
                }
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
            </label>

            <div className="space-y-1 text-sm">
              <span className="font-medium">Selected study</span>
              <div className="rounded-md border bg-muted/20 px-3 py-2 text-sm text-muted-foreground">
                {selectedStudyPath || "-"}
              </div>
            </div>
          </div>

          {studies.length === 0 && !loadingStudies ? (
            <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
              Run <code>python -m pipeline.cli physical-board-failure-study ...</code> and reload.
            </div>
          ) : null}
        </CardContent>
      </Card>

      {study ? (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Baseline board exact"
            value={formatPercent(study.report_metrics?.board_exact_match)}
            hint={study.summary.report_path ?? undefined}
          />
          <MetricCard
            label="Move recall"
            value={formatPercent(study.report_metrics?.move_detection_recall)}
            hint={`False-change ${formatPercent(study.report_metrics?.static_frame_false_change_rate)}`}
          />
          <MetricCard
            label="Selected / total"
            value={`${study.summary.selected_failures ?? 0} / ${study.summary.total_failures ?? 0}`}
            hint={study.summary.sample_mode ?? "-"}
          />
          <MetricCard
            label="Config"
            value={`${study.summary.config?.observation_input ?? "-"} · ${study.summary.config?.tracker_mode ?? "-"}`}
            hint={`lookahead w=${study.summary.config?.lookahead_window ?? "-"} m=${study.summary.config?.lookahead_margin ?? "-"}`}
          />
        </div>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
        <Card className="min-h-[640px]">
          <CardHeader>
            <CardTitle className="text-lg">Failures</CardTitle>
            <CardDescription>
              Filter by clip, bucket, or notes. The viewer always loads the selected row’s
              leading frames on demand.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search clip, bucket, note, move, frame…"
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
              <select
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={bucketFilter}
                onChange={(event) => setBucketFilter(event.target.value)}
              >
                {bucketOptions.map((option) => (
                  <option key={option} value={option}>
                    {option === "all" ? "All buckets" : option}
                  </option>
                ))}
              </select>
            </div>

            {study ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(study.bucket_counts).map(([bucket, count]) => (
                  <Badge key={bucket} variant={badgeVariantForBucket(bucket)}>
                    {bucket} · {count}
                  </Badge>
                ))}
              </div>
            ) : null}

            <div className="max-h-[760px] space-y-2 overflow-y-auto pr-1">
              {loadingStudy ? (
                <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  Loading study…
                </div>
              ) : filteredEntries.length === 0 ? (
                <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  No failures match the current filters.
                </div>
              ) : (
                filteredEntries.map((entry) => {
                  const selected = entry.selected_index === selectedIndex;
                  const bucket = bucketLabel(entry);
                  return (
                    <button
                      key={entry.selected_index}
                      type="button"
                      onClick={() => setSelectedIndex(entry.selected_index)}
                      className={`w-full rounded-lg border p-3 text-left transition-colors ${
                        selected
                          ? "border-foreground bg-muted/30"
                          : "border-border hover:bg-muted/20"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs text-muted-foreground">
                              #{entry.selected_index}
                            </span>
                            <span className="truncate text-sm font-medium">
                              {entry.clip_filename}
                            </span>
                          </div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            {formatFrameIndex(entry.frame_index)} · decoded {entry.decoded_error_count}
                            wrong · single {entry.stateless_error_count} wrong
                          </div>
                        </div>
                        <Badge variant={badgeVariantForBucket(bucket)} className="max-w-[180px] truncate">
                          {bucket}
                        </Badge>
                      </div>
                      <div className="mt-2 max-h-8 overflow-hidden text-xs text-muted-foreground">
                        {detailsText(entry)}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Context scrubber</CardTitle>
              <CardDescription>
                {selectedEntry
                  ? `${selectedEntry.clip_filename} · ${formatFrameIndex(selectedEntry.frame_index)}`
                  : "Select a failure from the list."}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {loadingContext ? (
                <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                  Loading context…
                </div>
              ) : !context || !currentFrame ? (
                <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                  Pick a failure to inspect its leading frames.
                </div>
              ) : (
                <>
                  <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-medium">
                            {formatFrameIndex(currentFrame.frame_index)}
                            {currentFrame.relative_offset === 0
                              ? " · anchor failure"
                              : ` · ${currentFrame.relative_offset}`}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {context.observation_input === "rectified_board"
                              ? "Rectified board crop"
                              : "Original oblique frame"}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={currentFrameOffset <= 0}
                            onClick={() => setCurrentFrameOffset((value) => Math.max(0, value - 1))}
                          >
                            Prev
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={currentFrameOffset >= context.frames.length - 1}
                            onClick={() =>
                              setCurrentFrameOffset((value) =>
                                Math.min(context.frames.length - 1, value + 1),
                              )
                            }
                          >
                            Next
                          </Button>
                        </div>
                      </div>

                      <img
                        src={currentFrame.image_data_url}
                        alt={`${context.entry.clip_filename} ${formatFrameIndex(currentFrame.frame_index)}`}
                        className="max-h-[560px] w-full rounded-lg border bg-muted/20 object-contain"
                      />

                      <div className="space-y-2">
                        <input
                          type="range"
                          min={0}
                          max={Math.max(0, context.frames.length - 1)}
                          step={1}
                          value={currentFrameOffset}
                          onChange={(event) => setCurrentFrameOffset(Number(event.target.value))}
                          className="w-full"
                        />
                        <div className="flex gap-2 overflow-x-auto pb-1">
                          {context.frames.map((frame, index) => (
                            <button
                              key={frame.annotation_id}
                              type="button"
                              onClick={() => setCurrentFrameOffset(index)}
                              className={`relative shrink-0 overflow-hidden rounded-md border ${
                                index === currentFrameOffset
                                  ? "border-foreground"
                                  : "border-border"
                              }`}
                            >
                              <img
                                src={frame.image_data_url}
                                alt={formatFrameIndex(frame.frame_index)}
                                className="h-20 w-20 object-cover"
                              />
                              <div className="absolute inset-x-0 bottom-0 bg-black/65 px-1 py-0.5 text-[10px] text-white">
                                {frame.relative_offset === 0 ? "anchor" : frame.relative_offset}
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="text-sm font-medium">Anchor diagnostics</div>
                        {context.anchor_panel_data_url ? (
                          <img
                            src={context.anchor_panel_data_url}
                            alt={`Diagnostics for ${context.entry.annotation_id}`}
                            className="w-full rounded-lg border bg-muted/20 object-contain"
                          />
                        ) : (
                          <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                            Rendered diagnostics panel unavailable for this entry.
                          </div>
                        )}
                      </div>

                      <div className="grid gap-2 text-sm sm:grid-cols-2">
                        <div className="rounded-md border p-3">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">
                            Suggested root cause
                          </div>
                          <div className="mt-1">{context.entry.suggested_root_cause ?? "-"}</div>
                        </div>
                        <div className="rounded-md border p-3">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">
                            Best legal move
                          </div>
                          <div className="mt-1">{context.entry.best_legal_move_uci ?? "-"}</div>
                        </div>
                        <div className="rounded-md border p-3">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">
                            GT legal rank
                          </div>
                          <div className="mt-1">{context.entry.gt_legal_rank ?? "-"}</div>
                        </div>
                        <div className="rounded-md border p-3">
                          <div className="text-xs uppercase tracking-wide text-muted-foreground">
                            Decoded move
                          </div>
                          <div className="mt-1">{context.entry.decoded_move_uci ?? "stay"}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Tagging</CardTitle>
              <CardDescription>
                Saves directly back to <code>manual_buckets.csv</code> for the selected failure.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto]">
                <div className="space-y-4">
                  <label className="space-y-1 text-sm">
                    <span className="font-medium">Final bucket</span>
                    <select
                      value={draftBucket}
                      onChange={(event) => setDraftBucket(event.target.value)}
                      className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                      disabled={!selectedEntry}
                    >
                      <option value="">(untagged)</option>
                      {(study?.bucket_options ?? []).map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="space-y-1 text-sm">
                    <span className="font-medium">Notes</span>
                    <textarea
                      value={draftNotes}
                      onChange={(event) => setDraftNotes(event.target.value)}
                      rows={4}
                      disabled={!selectedEntry}
                      className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                      placeholder="What actually failed here?"
                    />
                  </label>
                </div>

                <div className="flex items-end">
                  <Button type="button" onClick={() => void handleSave()} disabled={!selectedEntry || saving}>
                    {saving ? "Saving…" : "Save bucket"}
                  </Button>
                </div>
              </div>

              {context ? (
                <div className="grid gap-4 text-sm lg:grid-cols-2">
                  <div className="space-y-2 rounded-md border p-3">
                    <div className="font-medium">Anchor metadata</div>
                    <div className="text-muted-foreground">
                      {context.entry.clip_filename} · {formatFrameIndex(context.entry.frame_index)}
                    </div>
                    <div className="text-muted-foreground">
                      Decoded {context.entry.decoded_error_count} wrong · single {context.entry.stateless_error_count}
                      wrong
                    </div>
                    <div className="text-muted-foreground">
                      prevGT={String(Boolean(context.entry.decoded_matches_previous_gt))} · nextGT=
                      {String(Boolean(context.entry.decoded_matches_next_gt))} · bestLegalGT=
                      {String(Boolean(context.entry.best_legal_matches_gt))}
                    </div>
                  </div>
                  <div className="space-y-2 rounded-md border p-3">
                    <div className="font-medium">Square highlights</div>
                    <div className="text-muted-foreground">
                      GT changed: {(context.entry.gt_changed_squares ?? []).join(", ") || "-"}
                    </div>
                    <div className="text-muted-foreground">
                      Decoded changed: {(context.entry.decoded_changed_squares ?? []).join(", ") || "-"}
                    </div>
                    <div className="text-muted-foreground">
                      Decoded errors: {(context.entry.decoded_error_squares ?? []).join(", ") || "-"}
                    </div>
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
