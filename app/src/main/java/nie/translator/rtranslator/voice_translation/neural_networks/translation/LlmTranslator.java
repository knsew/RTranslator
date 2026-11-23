package nie.translator.rtranslator.voice_translation.neural_networks.translation;

import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;

/**
 * Translation through an external OpenAI-compatible LLM server
 * ({@code POST /v1/chat/completions}), used by the "Custom LLM" translation mode.
 * <p>
 * Only the translation step goes to the network: speech recognition (Whisper)
 * and text-to-speech stay on the device, exactly as with the local models.
 * <p>
 * The server is expected to accept the Qwen3/Halogen style request fields
 * {@code enable_thinking} and {@code reasoning_effort}; servers that ignore
 * unknown fields (OpenAI, llama.cpp, vLLM, ...) simply ignore them.
 * Thinking content is never shown to the user: it is either streamed as
 * {@code reasoning_content} (ignored here) or stripped from the final text
 * when it comes back inside {@code <think>...</think>} blocks.
 * <p>
 * When streaming is requested but the server does not support it, the request
 * is automatically retried once in non-streaming mode.
 */
public class LlmTranslator {
    private static final String TAG = "LlmTranslator";
    private static final int CONNECT_TIMEOUT_MS = 15000;
    private static final int READ_TIMEOUT_MS = 180000;
    private static final int MAX_TOKENS = 2048;

    //some servers return the thinking inside the answer text instead of a separate field
    private static final Pattern THINK_BLOCK = Pattern.compile("(?s)<think>.*?</think>");

    public interface PartialListener {
        void onPartial(String partialText);
    }

    /**
     * Translates a single text. When streaming is enabled and supported by the server,
     * partial results are delivered through {@code partialListener} as they arrive.
     *
     * @param baseUrl         server base URL, e.g. https://my-server:8000/v1 (the
     *                        /v1/chat/completions suffix may be included or omitted)
     * @param apiKey          API key, may be empty when the server needs none
     * @param model           model id as known by the server
     * @param thinkingEnabled when true the model may think before answering
     *                        (with a minimal reasoning effort, to limit latency);
     *                        when false thinking is explicitly disabled
     * @param stream          try server-sent events streaming first, with automatic
     *                        fallback to a single non-streaming request
     * @param sourceLanguage  source language name in English, e.g. "Chinese"
     * @param targetLanguage  target language name in English, e.g. "English"
     * @param text            text to translate
     * @param partialListener receives the growing translation while streaming, may be null
     * @return the translated text
     * @throws Exception on configuration errors or when both attempts fail
     */
    public static String translate(
            @NonNull String baseUrl,
            @Nullable String apiKey,
            @NonNull String model,
            boolean thinkingEnabled,
            boolean stream,
            @NonNull String sourceLanguage,
            @NonNull String targetLanguage,
            @NonNull String text,
            @Nullable PartialListener partialListener
    ) throws Exception {
        String chatUrl = resolveChatCompletionsUrl(baseUrl);
        JsonObject request = buildRequest(model, thinkingEnabled, stream, sourceLanguage, targetLanguage, text);

        if (stream) {
            try {
                return translateStreaming(chatUrl, apiKey, request, partialListener);
            } catch (Exception streamingException) {
                //streaming failed (server without SSE support, proxy dropping the stream, ...):
                //retry once with a single non-streaming request instead of failing outright
                Log.i(TAG, "Streaming translation failed, falling back to a non-streaming request: " + streamingException.getMessage());
                request = buildRequest(model, thinkingEnabled, false, sourceLanguage, targetLanguage, text);
            }
        }
        return translateOnce(chatUrl, apiKey, request);
    }

    /**
     * Checks that the server answers and returns how many models it lists.
     * Used by the settings dialog ("Test connection").
     */
    public static int testConnection(@NonNull String baseUrl, @Nullable String apiKey) throws Exception {
        String modelsUrl = normalizeBaseUrl(baseUrl) + "/models";
        HttpURLConnection connection = null;
        try {
            connection = openConnection(modelsUrl, apiKey, "application/json", CONNECT_TIMEOUT_MS, CONNECT_TIMEOUT_MS);
            connection.setRequestMethod("GET");
            int status = connection.getResponseCode();
            if (status != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP " + status + ": " + readAll(connection.getErrorStream()));
            }
            JsonObject response = JsonParser.parseString(readAll(connection.getInputStream())).getAsJsonObject();
            JsonArray data = response.getAsJsonArray("data");
            return data == null ? 0 : data.size();
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private static JsonObject buildRequest(String model, boolean thinkingEnabled, boolean stream,
                                           String sourceLanguage, String targetLanguage, String text) {
        JsonObject request = new JsonObject();
        request.addProperty("model", model);
        request.addProperty("stream", stream);
        request.addProperty("max_tokens", MAX_TOKENS);
        //always sent explicitly, so that "thinking off" really means off even on
        //servers whose default is to think (halogen defaults to thinking on, xhigh effort)
        request.addProperty("enable_thinking", thinkingEnabled);
        if (thinkingEnabled) {
            //cap the reasoning effort: translation rarely needs deep thinking,
            //and the default effort of some servers is very slow
            request.addProperty("reasoning_effort", "minimal");
        }

        JsonArray messages = new JsonArray();
        JsonObject systemMessage = new JsonObject();
        systemMessage.addProperty("role", "system");
        systemMessage.addProperty("content", "You are a translation engine. Translate the following text from "
                + sourceLanguage + " to " + targetLanguage
                + ". Output only the translated text, without explanations, quotes or extra formatting.");
        messages.add(systemMessage);
        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", text);
        messages.add(userMessage);
        request.add("messages", messages);
        return request;
    }

    private static String translateStreaming(String chatUrl, @Nullable String apiKey, JsonObject request,
                                             @Nullable PartialListener partialListener) throws Exception {
        HttpURLConnection connection = null;
        try {
            connection = openConnection(chatUrl, apiKey, "text/event-stream", CONNECT_TIMEOUT_MS, READ_TIMEOUT_MS);
            writeJsonBody(connection, request);
            int status = connection.getResponseCode();
            if (status != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP " + status + ": " + readAll(connection.getErrorStream()));
            }
            StringBuilder result = new StringBuilder();
            boolean sawEventStream = false;
            BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("data:")) {
                    continue;
                }
                String data = line.substring("data:".length()).trim();
                if (data.isEmpty()) {
                    continue;
                }
                sawEventStream = true;
                if ("[DONE]".equals(data)) {
                    break;
                }
                JsonObject chunk;
                try {
                    chunk = JsonParser.parseString(data).getAsJsonObject();
                } catch (Exception e) {
                    //ignore malformed chunks (keep-alive comments, proxies, ...)
                    continue;
                }
                JsonElement content = getDeltaContent(chunk);
                if (content != null && content.isJsonPrimitive()) {
                    String piece = content.getAsString();
                    if (!piece.isEmpty()) {
                        result.append(piece);
                        if (partialListener != null) {
                            partialListener.onPartial(result.toString());
                        }
                    }
                }
                //reasoning_content deltas are intentionally ignored
            }
            if (!sawEventStream) {
                throw new IOException("Server did not return a server-sent events stream");
            }
            return stripThinkBlocks(result.toString());
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private static String translateOnce(String chatUrl, @Nullable String apiKey, JsonObject request) throws Exception {
        HttpURLConnection connection = null;
        try {
            connection = openConnection(chatUrl, apiKey, "application/json", CONNECT_TIMEOUT_MS, READ_TIMEOUT_MS);
            writeJsonBody(connection, request);
            int status = connection.getResponseCode();
            if (status != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP " + status + ": " + readAll(connection.getErrorStream()));
            }
            JsonObject response = JsonParser.parseString(readAll(connection.getInputStream())).getAsJsonObject();
            JsonArray choices = response.getAsJsonArray("choices");
            if (choices == null || choices.size() == 0) {
                throw new IOException("Empty response from the server");
            }
            JsonObject message = choices.get(0).getAsJsonObject().getAsJsonObject("message");
            JsonElement content = message.get("content");
            if (content == null || content.isJsonNull()) {
                throw new IOException("Empty response from the server");
            }
            return stripThinkBlocks(content.getAsString());
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    private static HttpURLConnection openConnection(String url, @Nullable String apiKey, String accept,
                                                    int connectTimeoutMs, int readTimeoutMs) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        connection.setConnectTimeout(connectTimeoutMs);
        connection.setReadTimeout(readTimeoutMs);
        connection.setRequestProperty("Accept", accept);
        if (apiKey != null && !apiKey.trim().isEmpty()) {
            connection.setRequestProperty("Authorization", "Bearer " + apiKey.trim());
        }
        return connection;
    }

    private static void writeJsonBody(HttpURLConnection connection, JsonObject request) throws IOException {
        byte[] body = request.toString().getBytes(StandardCharsets.UTF_8);
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        connection.setFixedLengthStreamingMode(body.length);
        OutputStream output = connection.getOutputStream();
        try {
            output.write(body);
            output.flush();
        } finally {
            output.close();
        }
    }

    @Nullable
    private static JsonElement getDeltaContent(JsonObject chunk) {
        try {
            JsonArray choices = chunk.getAsJsonArray("choices");
            if (choices == null || choices.size() == 0) {
                return null;
            }
            JsonObject delta = choices.get(0).getAsJsonObject().getAsJsonObject("delta");
            if (delta == null) {
                return null;
            }
            return delta.get("content");
        } catch (Exception e) {
            return null;
        }
    }

    private static String stripThinkBlocks(String text) {
        if (text == null) {
            return "";
        }
        return THINK_BLOCK.matcher(text).replaceAll("").trim();
    }

    private static String readAll(@Nullable InputStream input) throws IOException {
        if (input == null) {
            return "";
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
        StringBuilder result = new StringBuilder();
        char[] buffer = new char[4096];
        int read;
        while ((read = reader.read(buffer)) != -1) {
            result.append(buffer, 0, read);
        }
        return result.toString();
    }

    /**
     * Normalizes a user-typed base URL: adds a scheme when missing
     * (http for local/private addresses, https otherwise), removes a trailing
     * slash and a trailing /chat/completions, if present.
     */
    static String normalizeBaseUrl(@NonNull String baseUrl) {
        String url = baseUrl.trim();
        while (url.endsWith("/")) {
            url = url.substring(0, url.length() - 1);
        }
        if (url.endsWith("/chat/completions")) {
            url = url.substring(0, url.length() - "/chat/completions".length());
        }
        if (!url.contains("://")) {
            url = (isLocalAddress(url) ? "http://" : "https://") + url;
        }
        return url;
    }

    static String resolveChatCompletionsUrl(@NonNull String baseUrl) {
        return normalizeBaseUrl(baseUrl) + "/chat/completions";
    }

    private static boolean isLocalAddress(String url) {
        String host = url.split("/")[0].split(":")[0].toLowerCase();
        if (host.equals("localhost") || host.equals("127.0.0.1")) {
            return true;
        }
        if (host.startsWith("10.") || host.startsWith("192.168.")) {
            return true;
        }
        if (host.startsWith("172.")) {
            try {
                int second = Integer.parseInt(host.split("\\.")[1]);
                return second >= 16 && second <= 31;
            } catch (Exception e) {
                return false;
            }
        }
        return false;
    }
}
