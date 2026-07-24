"""MusicBrainz lookup from disc TOC / disc ID."""

from __future__ import annotations

from dataclasses import dataclass, field

import musicbrainzngs

from .rip import DiscTOC, disc_id_from_toc

musicbrainzngs.set_useragent("SuperMapCD", "0.1.0", "https://github.com/local/supermap-cd-flac")


@dataclass
class TrackMeta:
    number: int
    title: str = "Unknown Title"
    recording_id: str | None = None


@dataclass
class AlbumMeta:
    discid: str
    album: str = "Unknown Album"
    artist: str = "Unknown Artist"
    release_id: str | None = None
    tracks: list[TrackMeta] = field(default_factory=list)

    def title_for(self, track_number: int) -> str:
        for t in self.tracks:
            if t.number == track_number:
                return t.title
        return f"Track {track_number:02d}"

    def recording_id_for(self, track_number: int) -> str | None:
        for t in self.tracks:
            if t.number == track_number:
                return t.recording_id
        return None


def lookup_disc(toc: DiscTOC) -> AlbumMeta:
    """Look up album metadata. Falls back to placeholders on failure."""
    discid = disc_id_from_toc(toc)
    meta = AlbumMeta(discid=discid)

    # Prefer real MusicBrainz discid library ID when available.
    try:
        import discid  # type: ignore

        disc = discid.read(toc.drive[0] + ":")
        discid = disc.id
        meta.discid = discid
        result = musicbrainzngs.get_releases_by_discid(
            discid,
            includes=["artists", "recordings"],
        )
    except Exception:
        # Without a valid MB discid, skip network lookup.
        for t in toc.toc_audio():
            meta.tracks.append(TrackMeta(number=t.number, title=f"Track {t.number:02d}"))
        return meta

    try:
        if "disc" in result:
            releases = result["disc"]["release-list"]
        elif "cdstub" in result:
            stub = result["cdstub"]
            meta.album = stub.get("title", meta.album)
            meta.artist = stub.get("artist", meta.artist)
            for i, t in enumerate(stub.get("track-list", []), start=1):
                meta.tracks.append(TrackMeta(number=i, title=t.get("title", f"Track {i:02d}")))
            return meta
        else:
            releases = []

        if not releases:
            for t in toc.toc_audio():
                meta.tracks.append(TrackMeta(number=t.number, title=f"Track {t.number:02d}"))
            return meta

        rel = releases[0]
        meta.release_id = rel.get("id")
        meta.album = rel.get("title", meta.album)
        artist_credit = rel.get("artist-credit", [])
        if artist_credit:
            meta.artist = artist_credit[0].get("artist", {}).get("name", meta.artist)

        # Medium track list matching this disc
        for medium in rel.get("medium-list", []):
            discs = medium.get("disc-list", [])
            if any(d.get("id") == discid for d in discs) or len(rel.get("medium-list", [])) == 1:
                for t in medium.get("track-list", []):
                    num = int(t.get("number", t.get("position", "0")) or 0)
                    recording = t.get("recording", {})
                    meta.tracks.append(
                        TrackMeta(
                            number=num,
                            title=recording.get("title", t.get("title", f"Track {num:02d}")),
                            recording_id=recording.get("id"),
                        )
                    )
                break

        if not meta.tracks:
            for t in toc.toc_audio():
                meta.tracks.append(TrackMeta(number=t.number, title=f"Track {t.number:02d}"))
    except Exception:
        for t in toc.toc_audio():
            meta.tracks.append(TrackMeta(number=t.number, title=f"Track {t.number:02d}"))

    return meta


def placeholder_meta(toc: DiscTOC) -> AlbumMeta:
    discid = disc_id_from_toc(toc)
    meta = AlbumMeta(discid=discid)
    for t in toc.toc_audio():
        meta.tracks.append(TrackMeta(number=t.number, title=f"Track {t.number:02d}"))
    return meta
