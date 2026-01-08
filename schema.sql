-- Create a table for Chat Sessions
create table chats (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users not null,
  title text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create a table for Messages
create table messages (
  id uuid default gen_random_uuid() primary key,
  chat_id uuid references chats(id) on delete cascade not null,
  sender text not null check (sender in ('user', 'ai')),
  content text,
  file_path text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Note: You must create a storage bucket named 'chat-files' in your Supabase Dashboard
-- and set it to private (not public).

-- Index for faster queries
create index chats_user_id_idx on chats(user_id);
create index messages_chat_id_idx on messages(chat_id);
create index messages_created_at_idx on messages(created_at);

-- Row Level Security (RLS)
alter table chats enable row level security;
alter table messages enable row level security;

-- Policy: Users can only see their own chats
create policy "Users can view their own chats"
  on chats for select
  using (auth.uid() = user_id);

create policy "Users can insert their own chats"
  on chats for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own chats"
  on chats for update
  using (auth.uid() = user_id);

create policy "Users can delete their own chats"
  on chats for delete
  using (auth.uid() = user_id);

-- Policy: Users can see messages in their chats
create policy "Users can view messages in their chats"
  on messages for select
  using (
    exists (
      select 1 from chats
      where chats.id = messages.chat_id
      and chats.user_id = auth.uid()
    )
  );

create policy "Users can insert messages into their chats"
  on messages for insert
  with check (
    exists (
      select 1 from chats
      where chats.id = messages.chat_id
      and chats.user_id = auth.uid()
    )
  );
