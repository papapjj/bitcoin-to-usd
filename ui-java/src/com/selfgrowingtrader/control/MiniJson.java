package com.selfgrowingtrader.control;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON parser for the append-only local telemetry stream.
 *
 * <p>The Java control app deliberately has no third-party runtime dependency,
 * so the live view parses only standard JSON here.  It accepts objects,
 * arrays, strings, numbers, booleans and null and preserves object order for
 * readable inspectors.</p>
 */
final class MiniJson {
    private MiniJson() {
    }

    static Object parse(String source) {
        if (source == null) {
            throw new IllegalArgumentException("JSON source is null");
        }
        Parser parser = new Parser(source);
        Object value = parser.readValue();
        parser.skipWhitespace();
        if (!parser.atEnd()) {
            throw parser.error("Unexpected trailing content");
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(Object value) {
        return value instanceof Map<?, ?> map ? (Map<String, Object>) map : Map.of();
    }

    @SuppressWarnings("unchecked")
    static List<Object> array(Object value) {
        return value instanceof List<?> list ? (List<Object>) list : List.of();
    }

    static String text(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value);
        return text.isBlank() ? fallback : text;
    }

    static double number(Object value, double fallback) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    static int integer(Object value, int fallback) {
        return (int) Math.round(number(value, fallback));
    }

    private static final class Parser {
        private final String source;
        private int index;

        private Parser(String source) {
            this.source = source;
        }

        private boolean atEnd() {
            return index >= source.length();
        }

        private void skipWhitespace() {
            while (!atEnd() && Character.isWhitespace(source.charAt(index))) {
                index++;
            }
        }

        private Object readValue() {
            skipWhitespace();
            if (atEnd()) {
                throw error("Expected a JSON value");
            }
            return switch (source.charAt(index)) {
                case '{' -> readObject();
                case '[' -> readArray();
                case '"' -> readString();
                case 't' -> readLiteral("true", Boolean.TRUE);
                case 'f' -> readLiteral("false", Boolean.FALSE);
                case 'n' -> readLiteral("null", null);
                default -> readNumber();
            };
        }

        private Map<String, Object> readObject() {
            expect('{');
            LinkedHashMap<String, Object> object = new LinkedHashMap<>();
            skipWhitespace();
            if (consume('}')) {
                return object;
            }
            while (true) {
                skipWhitespace();
                if (atEnd() || source.charAt(index) != '"') {
                    throw error("Expected an object key");
                }
                String key = readString();
                skipWhitespace();
                expect(':');
                Object value = readValue();
                object.put(key, value);
                skipWhitespace();
                if (consume('}')) {
                    return object;
                }
                expect(',');
            }
        }

        private List<Object> readArray() {
            expect('[');
            List<Object> array = new ArrayList<>();
            skipWhitespace();
            if (consume(']')) {
                return array;
            }
            while (true) {
                array.add(readValue());
                skipWhitespace();
                if (consume(']')) {
                    return array;
                }
                expect(',');
            }
        }

        private String readString() {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (!atEnd()) {
                char ch = source.charAt(index++);
                if (ch == '"') {
                    return out.toString();
                }
                if (ch != '\\') {
                    out.append(ch);
                    continue;
                }
                if (atEnd()) {
                    throw error("Incomplete string escape");
                }
                char escape = source.charAt(index++);
                switch (escape) {
                    case '"' -> out.append('"');
                    case '\\' -> out.append('\\');
                    case '/' -> out.append('/');
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'u' -> {
                        if (index + 4 > source.length()) {
                            throw error("Incomplete unicode escape");
                        }
                        String hex = source.substring(index, index + 4);
                        try {
                            out.append((char) Integer.parseInt(hex, 16));
                        } catch (NumberFormatException exc) {
                            throw error("Invalid unicode escape: " + hex);
                        }
                        index += 4;
                    }
                    default -> throw error("Unsupported string escape: \\" + escape);
                }
            }
            throw error("Unterminated string");
        }

        private Object readLiteral(String literal, Object value) {
            if (!source.startsWith(literal, index)) {
                throw error("Expected " + literal);
            }
            index += literal.length();
            return value;
        }

        private Number readNumber() {
            int start = index;
            if (consume('-')) {
                // sign consumed
            }
            if (consume('0')) {
                // zero consumed
            } else {
                if (atEnd() || !Character.isDigit(source.charAt(index))) {
                    throw error("Invalid number");
                }
                while (!atEnd() && Character.isDigit(source.charAt(index))) {
                    index++;
                }
            }
            if (consume('.')) {
                if (atEnd() || !Character.isDigit(source.charAt(index))) {
                    throw error("Invalid fractional number");
                }
                while (!atEnd() && Character.isDigit(source.charAt(index))) {
                    index++;
                }
            }
            if (!atEnd() && (source.charAt(index) == 'e' || source.charAt(index) == 'E')) {
                index++;
                if (!atEnd() && (source.charAt(index) == '+' || source.charAt(index) == '-')) {
                    index++;
                }
                if (atEnd() || !Character.isDigit(source.charAt(index))) {
                    throw error("Invalid exponent");
                }
                while (!atEnd() && Character.isDigit(source.charAt(index))) {
                    index++;
                }
            }
            String token = source.substring(start, index);
            try {
                if (token.indexOf('.') < 0 && token.indexOf('e') < 0 && token.indexOf('E') < 0) {
                    return Long.parseLong(token);
                }
                return Double.parseDouble(token);
            } catch (NumberFormatException exc) {
                throw error("Invalid number: " + token);
            }
        }

        private void expect(char expected) {
            skipWhitespace();
            if (atEnd() || source.charAt(index) != expected) {
                throw error("Expected '" + expected + "'");
            }
            index++;
        }

        private boolean consume(char expected) {
            if (!atEnd() && source.charAt(index) == expected) {
                index++;
                return true;
            }
            return false;
        }

        private IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at JSON offset " + index);
        }
    }
}
