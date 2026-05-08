package org.example.recruitmentservice.utils;

import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PositionUtils {
  public static String formatPositionTitle(String title, String seniority) {
    return Stream.of(seniority, title)
        .filter(Objects::nonNull)
        .map(String::trim)
        .filter(s -> !s.isBlank())
        .collect(Collectors.joining(" "));
  }
}
